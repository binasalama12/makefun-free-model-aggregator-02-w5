// FiveTry Failover Server — wire translation between the two client APIs and
// the single upstream API.
//
// Clients speak OpenAI Chat Completions or Anthropic Messages. The gateway
// only dials Chat Completions, so Anthropic requests are rewritten on the way
// up and answers are rewritten on the way back. The one invariant running
// through every function here: the `model` a client sees is the dial id this
// server actually sent upstream — providers echo the id back without the
// `openrouter/` prefix, so we stamp our own id onto every body and chunk.

import { bagOf, type Bag } from './brief.ts';

export interface ListedModel {
  dialId: string;
  vendor: string;
  label: string;
}

// Fixed stamps for the model listings: a listing that carried the wall clock
// would differ between a live run and its replay.
const LIST_CREATED_EPOCH = 1735689600;
const LIST_CREATED_ISO = '2025-01-01T00:00:00Z';

export function chatModelList(models: ListedModel[]): Bag {
  return {
    object: 'list',
    data: models.map((m) => ({
      id: m.dialId,
      object: 'model',
      created: LIST_CREATED_EPOCH,
      owned_by: m.vendor,
    })),
  };
}

export function messagesModelList(models: ListedModel[]): Bag {
  return {
    data: models.map((m) => ({
      type: 'model',
      id: m.dialId,
      display_name: m.label,
      created_at: LIST_CREATED_ISO,
    })),
    has_more: false,
    first_id: models.length > 0 ? models[0].dialId : null,
    last_id: models.length > 0 ? models[models.length - 1].dialId : null,
  };
}

export function chatError(message: string, type: string, code: string | null): Bag {
  return { error: { message, type, param: null, code } };
}

export function messagesError(type: string, message: string): Bag {
  return { type: 'error', error: { type, message } };
}

// --- request translation: Anthropic Messages -> Chat Completions ------------

function plainText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  const bits: string[] = [];
  for (const block of content) {
    if (bagOf(block) && block.type === 'text' && typeof block.text === 'string') bits.push(block.text);
  }
  return bits.join('\n');
}

function imagePart(block: Bag): Bag | null {
  if (block.type !== 'image' || !bagOf(block.source)) return null;
  const src = block.source;
  if (src.type === 'base64' && typeof src.data === 'string') {
    const media = typeof src.media_type === 'string' ? src.media_type : 'image/png';
    return { type: 'image_url', image_url: { url: `data:${media};base64,${src.data}` } };
  }
  if (src.type === 'url' && typeof src.url === 'string') {
    return { type: 'image_url', image_url: { url: src.url } };
  }
  return null;
}

function toolResultAsText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const t = plainText(content);
    return t !== '' ? t : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** Rewrite a Messages request as the Chat Completions body the gateway dials. */
export function messagesToChat(body: Bag, dialId: string): Bag {
  const msgs: Bag[] = [];
  const sys = plainText(body.system);
  if (sys !== '') msgs.push({ role: 'system', content: sys });

  const incoming = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of incoming) {
    if (!bagOf(raw)) continue;
    const isAssistant = raw.role === 'assistant';
    if (typeof raw.content === 'string') {
      msgs.push({ role: isAssistant ? 'assistant' : 'user', content: raw.content });
      continue;
    }
    if (!Array.isArray(raw.content)) continue;
    if (isAssistant) {
      const calls: Bag[] = [];
      for (const block of raw.content) {
        if (!bagOf(block) || block.type !== 'tool_use') continue;
        calls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(bagOf(block.input) ? block.input : {}),
          },
        });
      }
      const text = plainText(raw.content);
      const msg: Bag = { role: 'assistant', content: text === '' ? null : text };
      if (calls.length > 0) msg.tool_calls = calls;
      msgs.push(msg);
      continue;
    }
    const parts: Bag[] = [];
    for (const block of raw.content) {
      if (!bagOf(block)) continue;
      if (block.type === 'tool_result') {
        msgs.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultAsText(block.content),
        });
        continue;
      }
      if (block.type === 'text' && typeof block.text === 'string') {
        parts.push({ type: 'text', text: block.text });
        continue;
      }
      const img = imagePart(block);
      if (img !== null) parts.push(img);
    }
    if (parts.length === 1 && parts[0].type === 'text') {
      msgs.push({ role: 'user', content: parts[0].text as string });
    } else if (parts.length > 0) {
      msgs.push({ role: 'user', content: parts });
    }
  }

  const out: Bag = { model: dialId, messages: msgs };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (typeof body.top_k === 'number') out.top_k = body.top_k;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools: Bag[] = [];
    for (const rawTool of body.tools) {
      if (!bagOf(rawTool) || typeof rawTool.name !== 'string') continue;
      tools.push({
        type: 'function',
        function: {
          name: rawTool.name,
          description: typeof rawTool.description === 'string' ? rawTool.description : '',
          parameters: bagOf(rawTool.input_schema) ? rawTool.input_schema : { type: 'object', properties: {} },
        },
      });
    }
    if (tools.length > 0) out.tools = tools;
  }
  if (bagOf(body.tool_choice)) {
    const tc = body.tool_choice;
    if (tc.type === 'auto') out.tool_choice = 'auto';
    else if (tc.type === 'any') out.tool_choice = 'required';
    else if (tc.type === 'none') out.tool_choice = 'none';
    else if (tc.type === 'tool' && typeof tc.name === 'string') {
      out.tool_choice = { type: 'function', function: { name: tc.name } };
    }
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

// --- answer translation: Chat Completions -> Anthropic Messages --------------

const numOr0 = (v: unknown): number => (typeof v === 'number' && Number.isFinite(v) ? v : 0);

export function finishToStopReason(finish: unknown): string {
  switch (finish) {
    case 'length':
      return 'max_tokens';
    case 'tool_calls':
    case 'function_call':
      return 'tool_use';
    case 'content_filter':
      return 'refusal';
    case 'stop':
    default:
      return 'end_turn';
  }
}

function argsToObject(value: unknown): Bag {
  if (bagOf(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (bagOf(parsed)) return parsed;
    } catch {
      // unparseable arguments become an empty input object
    }
  }
  return {};
}

/** Shape a Chat Completions JSON answer as a Messages object naming `dialId`. */
export function chatToMessages(body: Bag, dialId: string): Bag {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const first = choices.length > 0 && bagOf(choices[0]) ? choices[0] : {};
  const msg = bagOf(first.message) ? first.message : {};
  const blocks: Bag[] = [];
  const text = typeof msg.content === 'string' ? msg.content : plainText(msg.content);
  if (text !== '') blocks.push({ type: 'text', text });
  const calls = Array.isArray(msg.tool_calls) ? msg.tool_calls : [];
  for (const call of calls) {
    if (!bagOf(call) || !bagOf(call.function)) continue;
    blocks.push({
      type: 'tool_use',
      id: typeof call.id === 'string' && call.id !== '' ? call.id : 'toolu_fivetry',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      input: argsToObject(call.function.arguments),
    });
  }
  if (blocks.length === 0) blocks.push({ type: 'text', text: '' });
  const usage = bagOf(body.usage) ? body.usage : {};
  const rawId = typeof body.id === 'string' ? body.id.replace(/[^A-Za-z0-9_-]/g, '') : 'answer';
  return {
    id: `msg_${rawId}`,
    type: 'message',
    role: 'assistant',
    model: dialId,
    content: blocks,
    stop_reason: finishToStopReason(first.finish_reason),
    stop_sequence: null,
    usage: {
      input_tokens: numOr0(usage.prompt_tokens),
      output_tokens: numOr0(usage.completion_tokens),
    },
  };
}

/** Stamp the dialled id onto a Chat Completions JSON answer. */
export function stampChatJson(body: Bag, dialId: string): Bag {
  return { ...body, model: dialId };
}

// --- streaming ------------------------------------------------------------------

export function sseFrame(eventName: string | null, data: unknown): string {
  const head = eventName === null ? '' : `event: ${eventName}\n`;
  return `${head}data: ${JSON.stringify(data)}\n\n`;
}

/** One raw SSE frame with every JSON payload's model field replaced. */
function restampFrame(frame: string, dialId: string): string {
  const lines = frame.split('\n').map((line) => {
    if (!line.startsWith('data:')) return line;
    const payload = line.slice(5).trim();
    if (payload === '' || payload === '[DONE]') return line;
    try {
      const chunk: unknown = JSON.parse(payload);
      if (bagOf(chunk)) return `data: ${JSON.stringify({ ...chunk, model: dialId })}`;
    } catch {
      // not JSON: pass the line through untouched
    }
    return line;
  });
  return `${lines.join('\n')}\n\n`;
}

/** Split a byte stream into whole SSE frames (text between blank lines). */
async function* framesOf(stream: ReadableStream<Uint8Array>): AsyncGenerator<string> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let pending = '';
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      pending += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let cut = pending.indexOf('\n\n');
      while (cut !== -1) {
        yield pending.slice(0, cut);
        pending = pending.slice(cut + 2);
        cut = pending.indexOf('\n\n');
      }
    }
    pending += decoder.decode();
    if (pending.trim() !== '') yield pending;
  } finally {
    reader.releaseLock();
  }
}

/** Relay a Chat Completions SSE stream, restamping the model on every chunk. */
export async function* relayChatSse(
  stream: ReadableStream<Uint8Array>,
  dialId: string,
): AsyncGenerator<string> {
  for await (const frame of framesOf(stream)) {
    yield restampFrame(frame, dialId);
  }
}

/**
 * Relay a Chat Completions SSE stream as the Anthropic event sequence:
 * message_start, content blocks (text and tool_use) with their deltas, then
 * message_delta carrying the stop reason and output usage, then message_stop.
 * The finish reason is held back until the trailing usage chunk has been
 * seen, because Anthropic's message_delta wants both together.
 */
export async function* relayMessagesSse(
  stream: ReadableStream<Uint8Array>,
  dialId: string,
): AsyncGenerator<string> {
  let announced = false;
  let sealed = false;
  let heldStop: string | null = null;
  let inTokens = 0;
  let outTokens = 0;
  let openTextAt = -1;
  let nextSlot = 0;
  const toolSlots = new Map<number, number>();

  const announce = (): string[] => {
    if (announced) return [];
    announced = true;
    return [
      sseFrame('message_start', {
        type: 'message_start',
        message: {
          id: 'msg_fivetry_stream',
          type: 'message',
          role: 'assistant',
          model: dialId,
          content: [],
          stop_reason: null,
          stop_sequence: null,
          usage: { input_tokens: inTokens, output_tokens: 0 },
        },
      }),
    ];
  };

  const closeBlocks = (): string[] => {
    const out: string[] = [];
    if (openTextAt !== -1) {
      out.push(sseFrame('content_block_stop', { type: 'content_block_stop', index: openTextAt }));
      openTextAt = -1;
    }
    const slots = [...toolSlots.values()].sort((a, b) => a - b);
    for (const slot of slots) out.push(sseFrame('content_block_stop', { type: 'content_block_stop', index: slot }));
    toolSlots.clear();
    return out;
  };

  const seal = (reason: string): string[] => {
    if (sealed) return [];
    sealed = true;
    return [
      ...closeBlocks(),
      sseFrame('message_delta', {
        type: 'message_delta',
        delta: { stop_reason: reason, stop_sequence: null },
        usage: { output_tokens: outTokens },
      }),
      sseFrame('message_stop', { type: 'message_stop' }),
    ];
  };

  for await (const frame of framesOf(stream)) {
    for (const line of frame.split('\n')) {
      if (!line.startsWith('data:')) continue;
      const payload = line.slice(5).trim();
      if (payload === '[DONE]') {
        for (const e of announce()) yield e;
        for (const e of seal(heldStop ?? 'end_turn')) yield e;
        continue;
      }
      let chunk: unknown;
      try {
        chunk = JSON.parse(payload);
      } catch {
        continue;
      }
      if (!bagOf(chunk)) continue;
      if (bagOf(chunk.usage)) {
        inTokens = numOr0(chunk.usage.prompt_tokens);
        outTokens = numOr0(chunk.usage.completion_tokens);
      }
      const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
      const choice = choices.length > 0 && bagOf(choices[0]) ? choices[0] : null;
      if (choice === null) continue;
      for (const e of announce()) yield e;
      const delta = bagOf(choice.delta) ? choice.delta : {};
      if (typeof delta.content === 'string' && delta.content !== '') {
        if (openTextAt === -1) {
          openTextAt = nextSlot;
          nextSlot += 1;
          yield sseFrame('content_block_start', {
            type: 'content_block_start',
            index: openTextAt,
            content_block: { type: 'text', text: '' },
          });
        }
        yield sseFrame('content_block_delta', {
          type: 'content_block_delta',
          index: openTextAt,
          delta: { type: 'text_delta', text: delta.content },
        });
      }
      const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
      for (const call of calls) {
        if (!bagOf(call)) continue;
        const upAt = typeof call.index === 'number' ? call.index : toolSlots.size;
        let slot = toolSlots.get(upAt);
        if (slot === undefined) {
          if (openTextAt !== -1) {
            yield sseFrame('content_block_stop', { type: 'content_block_stop', index: openTextAt });
            openTextAt = -1;
          }
          slot = nextSlot;
          nextSlot += 1;
          toolSlots.set(upAt, slot);
          const fn = bagOf(call.function) ? call.function : {};
          yield sseFrame('content_block_start', {
            type: 'content_block_start',
            index: slot,
            content_block: {
              type: 'tool_use',
              id: typeof call.id === 'string' && call.id !== '' ? call.id : `toolu_${slot}`,
              name: typeof fn.name === 'string' ? fn.name : '',
              input: {},
            },
          });
        }
        const fn = bagOf(call.function) ? call.function : {};
        if (typeof fn.arguments === 'string' && fn.arguments !== '') {
          yield sseFrame('content_block_delta', {
            type: 'content_block_delta',
            index: slot,
            delta: { type: 'input_json_delta', partial_json: fn.arguments },
          });
        }
      }
      if (choice.finish_reason !== null && choice.finish_reason !== undefined) {
        heldStop = finishToStopReason(choice.finish_reason);
      }
    }
  }
  if (!sealed) {
    for (const e of announce()) yield e;
    for (const e of seal(heldStop ?? 'end_turn')) yield e;
  }
}
