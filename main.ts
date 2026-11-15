// FiveTry Failover Server — a free-model aggregator behind both client wires.
//
// One process, three jobs:
//   1. read the trial brief, build the free catalog from it, print that exact
//      document for `--list`, and register it with the env;
//   2. serve GET /v1/models plus POST /v1/chat/completions and
//      POST /v1/messages, answering "auto" with the best-rated free model
//      that fits the request, dialled through the gateway;
//   3. when a dial is refused (429, 5xx, or the network itself), bench that
//      model and try the next fit — up to five dials per request — and carry
//      the refusal trail on the final answer's x-makefun-upstream-failures
//      header whenever anything was refused along the way.
//
// Determinism: routing, cooldowns and ordering are functions of the brief,
// the request sequence and the gateway's answers only. The cooldown clock is
// a logical request counter; nothing here reads the wall clock or a random
// source, so a replay walks the same path as the live run.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  bagOf,
  catalogFromBrief,
  freeFleet,
  loadBrief,
  type Bag,
  type CatalogDoc,
  type NeedSet,
} from './lib/brief.ts';
import {
  MAX_TRIES,
  SitOutBook,
  candidateQueue,
  mergeNeeds,
  needsFromChatBody,
  needsFromMessagesBody,
} from './lib/router.ts';
import {
  chatError,
  chatModelList,
  chatToMessages,
  messagesError,
  messagesModelList,
  messagesToChat,
  relayChatSse,
  relayMessagesSse,
  stampChatJson,
} from './lib/translate.ts';

const PROTOCOL_VERSION = 1;
const WORK_POLL_MS = 100;
const BODY_CAP_BYTES = 8 * 1024 * 1024;
const FALLBACK_MAX_TOKENS = 1024;
const TOKEN_CEILING = 8192;

const ENV_BASE = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const BRIEF_PATH = process.env.MAKEFUN_PROMPT_FILE ?? '';
const GATEWAY_BASE = (process.env.OPENAI_BASE_URL ?? '').replace(/\/+$/, '');
const GATEWAY_KEY = process.env.OPENAI_API_KEY ?? '';

const brief = loadBrief(BRIEF_PATH);
const catalogDoc: CatalogDoc = catalogFromBrief(brief);
const catalogText = JSON.stringify(catalogDoc);
const fleet = freeFleet(brief);
const sitOuts = new SitOutBook();

/** Logical clock: counts completion requests served so far. */
let requestTick = 0;
/** Manifest cursor: the brief lists prompts in the order the env sends them. */
let manifestCursor = 0;

function note(msg: string): void {
  process.stderr.write(`fivetry: ${msg}\n`);
}

function gatewayChatUrl(): string {
  if (GATEWAY_BASE === '') return '';
  return GATEWAY_BASE.endsWith('/v1')
    ? `${GATEWAY_BASE}/chat/completions`
    : `${GATEWAY_BASE}/v1/chat/completions`;
}

interface Refusal {
  status: number;
  code: string;
  model: string | null;
}

function refusalHeader(refusals: Refusal[]): Record<string, string> {
  if (refusals.length === 0) return {};
  return { 'x-makefun-upstream-failures': JSON.stringify(refusals.slice(0, 8)) };
}

/** Pull a machine code out of a gateway error body, for the refusal trail. */
function codeFromErrorText(text: string, status: number): string {
  try {
    const parsed: unknown = JSON.parse(text);
    if (bagOf(parsed) && bagOf(parsed.error)) {
      const err = parsed.error;
      if (typeof err.code === 'string' && err.code !== '') return err.code;
      if (typeof err.type === 'string' && err.type !== '') return err.type;
    }
  } catch {
    // not JSON: fall through to the status-based code
  }
  return `http_${status}`;
}

async function readJsonBody(req: IncomingMessage): Promise<Bag | null> {
  const parts: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buf = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buf.length;
    if (size > BODY_CAP_BYTES) return null;
    parts.push(buf);
  }
  if (parts.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(parts).toString('utf8'));
    return bagOf(value) ? value : null;
  } catch {
    return null;
  }
}

function replyJson(
  res: ServerResponse,
  status: number,
  body: unknown,
  extraHeaders: Record<string, string> = {},
): void {
  const text = JSON.stringify(body);
  res.writeHead(status, {
    'content-type': 'application/json',
    'content-length': Buffer.byteLength(text),
    ...extraHeaders,
  });
  res.end(text);
}

function replyError(
  res: ServerResponse,
  wire: 'openai' | 'anthropic',
  status: number,
  type: string,
  message: string,
  extraHeaders: Record<string, string> = {},
): void {
  if (wire === 'anthropic') replyJson(res, status, messagesError(type, message), extraHeaders);
  else replyJson(res, status, chatError(message, type, null), extraHeaders);
}

/** The manifest row for the request arriving now, advancing the cursor. */
function manifestNeeds(): NeedSet {
  const row = brief.manifest[manifestCursor];
  manifestCursor += 1;
  return row ? { ...row.needs } : {};
}

/** Keep every upstream body inside the gateway's output-cap rule. */
function enforceTokenCap(body: Bag): Bag {
  const out = { ...body };
  const cap =
    typeof out.max_completion_tokens === 'number'
      ? out.max_completion_tokens
      : typeof out.max_tokens === 'number'
        ? out.max_tokens
        : null;
  if (cap === null) {
    out.max_tokens = FALLBACK_MAX_TOKENS;
  } else if (cap > TOKEN_CEILING) {
    if (typeof out.max_completion_tokens === 'number') out.max_completion_tokens = TOKEN_CEILING;
    else out.max_tokens = TOKEN_CEILING;
  }
  return out;
}

type DialOutcome =
  | { kind: 'ok'; response: Response; dialId: string }
  | { kind: 'gave-up'; lastStatus: number; lastText: string };

/**
 * The FiveTry loop: walk the candidate queue, dial each model once, bench it
 * on a retryable refusal and move on, for at most MAX_TRIES dials. A
 * non-retryable refusal (a 4xx that is not 429) stops the walk at once —
 * re-dialling a different model cannot fix a malformed request.
 */
async function dialWithFailover(
  queue: { dialId: string }[],
  baseBody: Bag,
  wantsStream: boolean,
  tick: number,
  refusals: Refusal[],
): Promise<DialOutcome> {
  let lastStatus = 502;
  let lastText = '';
  const attempts = Math.min(MAX_TRIES, queue.length);
  for (let i = 0; i < attempts; i += 1) {
    const dialId = queue[i].dialId;
    const payload = enforceTokenCap({ ...baseBody, model: dialId });
    if (wantsStream) {
      payload.stream = true;
      payload.stream_options = { include_usage: true };
    }
    let response: Response;
    try {
      response = await fetch(gatewayChatUrl(), {
        method: 'POST',
        headers: {
          'content-type': 'application/json',
          authorization: `Bearer ${GATEWAY_KEY}`,
          ...(wantsStream ? { accept: 'text/event-stream' } : {}),
        },
        body: JSON.stringify(payload),
      });
    } catch (err) {
      refusals.push({ status: 0, code: 'network_error', model: dialId });
      sitOuts.bench(dialId, tick, 0);
      lastStatus = 502;
      lastText = (err as Error).message;
      continue;
    }
    if (response.ok && response.body !== null) {
      sitOuts.forgive(dialId);
      return { kind: 'ok', response, dialId };
    }
    const text = await response.text().catch(() => '');
    const code = codeFromErrorText(text, response.status);
    refusals.push({ status: response.status, code, model: dialId });
    sitOuts.bench(dialId, tick, response.status);
    lastStatus = response.status;
    lastText = text;
    const retryable = response.status === 429 || response.status >= 500;
    if (!retryable) break;
  }
  return { kind: 'gave-up', lastStatus, lastText };
}

/** Shared completion path for both wires. */
async function serveCompletion(
  wire: 'openai' | 'anthropic',
  req: IncomingMessage,
  res: ServerResponse,
): Promise<void> {
  const body = await readJsonBody(req);
  if (body === null) {
    replyError(res, wire, 400, 'invalid_request_error', 'request body must be one JSON object');
    return;
  }
  requestTick += 1;
  const tick = requestTick;
  const wantsStream = body.stream === true;
  const sniffed = wire === 'openai' ? needsFromChatBody(body) : needsFromMessagesBody(body);
  const needs = mergeNeeds(manifestNeeds(), sniffed);
  const asked = typeof body.model === 'string' ? body.model : '';
  const pinned = asked !== '' && asked !== 'auto' ? asked : null;
  const queue = candidateQueue(fleet, needs, sitOuts, tick, pinned);
  if (queue.length === 0) {
    replyError(
      res,
      wire,
      503,
      wire === 'anthropic' ? 'overloaded_error' : 'no_free_model',
      'no free model in the brief fits this request',
    );
    return;
  }
  // The upstream body: the client's own body on the OpenAI wire (minus the
  // placeholder model name), a translation on the Anthropic wire.
  const baseBody: Bag =
    wire === 'openai'
      ? (({ model: _drop, ...rest }) => rest)(body)
      : (({ model: _drop, stream: _s, ...rest }) => messagesToChat({ ...rest, stream: false }, ''))(body);

  const refusals: Refusal[] = [];
  const outcome = await dialWithFailover(queue, baseBody, wantsStream, tick, refusals);
  const trail = refusalHeader(refusals);
  if (outcome.kind === 'gave-up') {
    replyError(
      res,
      wire,
      502,
      wire === 'anthropic' ? 'api_error' : 'upstream_error',
      `gateway refused on every try (last HTTP ${outcome.lastStatus})${outcome.lastText === '' ? '' : `: ${outcome.lastText.slice(0, 400)}`}`,
      trail,
    );
    return;
  }
  const { response, dialId } = outcome;
  if (wantsStream) {
    res.writeHead(200, {
      'content-type': 'text/event-stream',
      'cache-control': 'no-cache',
      connection: 'keep-alive',
      ...trail,
    });
    try {
      const relay = wire === 'openai' ? relayChatSse(response.body!, dialId) : relayMessagesSse(response.body!, dialId);
      for await (const piece of relay) res.write(piece);
    } finally {
      res.end();
    }
    return;
  }
  const rawText = await response.text();
  let answer: Bag = {};
  try {
    const parsed: unknown = JSON.parse(rawText);
    if (bagOf(parsed)) answer = parsed;
  } catch {
    // an ok status with an unparseable body is relayed as an empty answer
  }
  if (wire === 'openai') replyJson(res, 200, stampChatJson(answer, dialId), trail);
  else replyJson(res, 200, chatToMessages(answer, dialId), trail);
}

function serveModelList(req: IncomingMessage, res: ServerResponse): void {
  const listed = fleet.map((m) => ({ dialId: m.dialId, vendor: m.vendor, label: m.label }));
  if (typeof req.headers['x-api-key'] === 'string') replyJson(res, 200, messagesModelList(listed));
  else replyJson(res, 200, chatModelList(listed));
}

async function route(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://fivetry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    serveModelList(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await serveCompletion('openai', req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await serveCompletion('anthropic', req, res);
    return;
  }
  replyError(
    res,
    url.pathname === '/v1/messages' ? 'anthropic' : 'openai',
    404,
    'not_found',
    `no route for ${req.method ?? ''} ${url.pathname}`,
  );
}

function listenOnFreePort(server: Server): Promise<number> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => {
      const addr = server.address();
      if (addr === null || typeof addr === 'string') reject(new Error('server picked no port'));
      else resolve(addr.port);
    });
  });
}

async function registerWithEnv(port: number): Promise<void> {
  const res = await fetch(`${ENV_BASE}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    // The catalog goes over as the parsed document of catalogText, so the
    // registered bytes and the --list bytes describe the identical object.
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: JSON.parse(catalogText) }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) {
    throw new Error(`register answered HTTP ${res.status}: ${(await res.text()).slice(0, 400)}`);
  }
  const reply: unknown = await res.json().catch(() => null);
  if (bagOf(reply) && typeof reply.requests === 'number') {
    note(`registered on port ${port}; trial holds ${reply.requests} prompts`);
  }
}

/** Poll the env's /work until it says the trial is done, then exit 0. */
function watchWork(server: Server): void {
  let busy = false;
  const timer = setInterval(() => {
    if (busy) return;
    busy = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_BASE}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status === 200) {
          const work: unknown = await res.json().catch(() => null);
          if (bagOf(work) && work.done === true) {
            clearInterval(timer);
            server.close();
            process.exit(0);
          }
        }
      } catch {
        // env not answering yet: the next tick retries
      } finally {
        busy = false;
      }
    })();
  }, WORK_POLL_MS);
}

async function boot(): Promise<void> {
  if (brief.problem !== null) note(`brief warning: ${brief.problem}`);
  if (process.argv.includes('--list')) {
    await new Promise<void>((resolve) => process.stdout.write(`${catalogText}\n`, () => resolve()));
    process.exit(0);
  }
  if (ENV_BASE === '') throw new Error('MAKEFUN_ENV_URL is not set');
  if (gatewayChatUrl() === '') throw new Error('OPENAI_BASE_URL is not set');
  const server = createServer((req, res) => {
    void route(req, res).catch((err: unknown) => {
      note((err as Error).stack ?? String(err));
      if (!res.headersSent) {
        replyError(res, 'openai', 500, 'internal_error', 'the server failed while answering');
      } else if (!res.writableEnded) {
        res.end();
      }
    });
  });
  const port = await listenOnFreePort(server);
  if (port === 8080) throw new Error('the OS handed us 8080, which belongs to the env: refusing to start');
  note(`FiveTry Failover Server listening on 0.0.0.0:${port} with ${fleet.length} free models`);
  await registerWithEnv(port);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  watchWork(server);
}

try {
  await boot();
} catch (err) {
  note((err as Error).stack ?? String(err));
  process.exit(1);
}
