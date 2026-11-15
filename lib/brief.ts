// FiveTry Failover Server — brief reading and catalog building.
//
// The trial brief is a prose file with two JSON payloads buried in it. We do
// not care about the prose at all: we scan for balanced {...} blocks, keep the
// ones that parse and carry a `schema` string from this contest's family, and
// read the model table plus the prompt manifest out of them. Everything in
// this module is a pure function of the file bytes, so the catalog printed by
// `--list` and the catalog sent at registration can never drift apart.

import { readFileSync } from 'node:fs';

export type Bag = Record<string, unknown>;

export const bagOf = (v: unknown): v is Bag =>
  typeof v === 'object' && v !== null && !Array.isArray(v);

const FAMILY = 'free-model-aggregator-02/';
export const PUBLIC_BASE = 'https://openrouter.ai/api/v1';
export const KEY_ENV_NAME = 'OPENROUTER_API_KEY';

export interface FreeModel {
  dialId: string;
  vendor: string;
  label: string;
  scoreMilli: number;
  windowTokens: number | null;
  canTools: boolean | null;
  canVision: boolean | null;
  canJson: boolean | null;
  isFree: boolean;
}

export interface NeedSet {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  minWindow?: number;
}

export interface PromptSpec {
  tag: string;
  wire: 'openai' | 'anthropic';
  wantsStream: boolean;
  weight: number;
  needs: NeedSet;
}

export interface TrialBrief {
  fleet: FreeModel[];
  manifest: PromptSpec[];
  problem: string | null;
}

export interface CatalogDoc {
  providers: Array<{
    id: string;
    base_url: string;
    key_env: string;
    wire: 'openai';
    models: Array<{ id: string; free: boolean; context_length: number | null }>;
  }>;
}

// --- scanning ---------------------------------------------------------------

/** Index just past the brace that closes the object opened at `openAt`. */
function closingBrace(text: string, openAt: number): number {
  let depth = 0;
  let quoted = false;
  for (let i = openAt; i < text.length; i += 1) {
    const c = text[i];
    if (quoted) {
      if (c === '\\') i += 1;
      else if (c === '"') quoted = false;
    } else if (c === '"') quoted = true;
    else if (c === '{') depth += 1;
    else if (c === '}') {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

/** Every parseable JSON object in the brief that declares a contest schema. */
function schemaPayloads(text: string): Bag[] {
  const found: Bag[] = [];
  for (let at = text.indexOf('{'); at !== -1; at = text.indexOf('{', at + 1)) {
    const end = closingBrace(text, at);
    if (end < 0) continue;
    try {
      const parsed: unknown = JSON.parse(text.slice(at, end + 1));
      if (bagOf(parsed) && typeof parsed.schema === 'string' && parsed.schema.includes(FAMILY)) {
        found.push(parsed);
      }
    } catch {
      // prose brace or partial JSON: keep scanning
    }
  }
  return found;
}

function firstArrayPayload(payloads: Bag[], keys: string[]): unknown[] | null {
  for (const p of payloads) {
    for (const k of keys) {
      if (Array.isArray(p[k])) return p[k] as unknown[];
    }
  }
  return null;
}

// --- row shaping --------------------------------------------------------------

const asText = (v: unknown): string => (typeof v === 'string' ? v : '');
const asNum = (v: unknown): number | null =>
  typeof v === 'number' && Number.isFinite(v) ? v : null;
const asBool = (v: unknown): boolean | null => (typeof v === 'boolean' ? v : null);

function shapeModel(raw: unknown): FreeModel | null {
  if (!bagOf(raw)) return null;
  let dialId = asText(raw.id) || asText(raw.model);
  if (dialId === '') return null;
  const vendor = asText(raw.provider) || asText(raw.wire) || 'openrouter';
  // A bare upstream slug under the openrouter vendor still needs the prefix
  // the gateway routes on.
  if (vendor === 'openrouter' && !dialId.startsWith('openrouter/')) {
    dialId = `openrouter/${dialId}`;
  }
  return {
    dialId,
    vendor,
    label: asText(raw.display_name) || dialId,
    scoreMilli: asNum(raw.rating_milli) ?? 0,
    windowTokens: asNum(raw.context_length),
    canTools: asBool(raw.tools),
    canVision: asBool(raw.vision),
    canJson: asBool(raw.json),
    isFree: raw.free !== false,
  };
}

function shapePrompt(raw: unknown): PromptSpec | null {
  if (!bagOf(raw)) return null;
  const tag = asText(raw.id);
  if (tag === '') return null;
  const req = bagOf(raw.requires) ? raw.requires : {};
  const needs: NeedSet = {};
  if (req.tools === true) needs.tools = true;
  if (req.vision === true) needs.vision = true;
  if (req.json === true) needs.json = true;
  const mc = asNum(req.min_context);
  if (mc !== null) needs.minWindow = mc;
  return {
    tag,
    wire: raw.api === 'anthropic' ? 'anthropic' : 'openai',
    wantsStream: raw.stream === true,
    weight: raw.weight === 2 ? 2 : 1,
    needs,
  };
}

/** Load and shape the brief; problems are reported, never thrown. */
export function loadBrief(filePath: string): TrialBrief {
  if (!filePath) return { fleet: [], manifest: [], problem: 'MAKEFUN_PROMPT_FILE missing' };
  let text: string;
  try {
    text = readFileSync(filePath, 'utf8');
  } catch (err) {
    return { fleet: [], manifest: [], problem: `unreadable brief: ${(err as Error).message}` };
  }
  const payloads = schemaPayloads(text);
  const modelRows = firstArrayPayload(payloads, ['models', 'ratings']);
  if (modelRows === null) return { fleet: [], manifest: [], problem: 'brief has no model table' };
  const fleet = modelRows.map(shapeModel).filter((m): m is FreeModel => m !== null);
  const promptRows = firstArrayPayload(payloads, ['prompts', 'manifest']);
  if (promptRows === null) return { fleet, manifest: [], problem: 'brief has no prompt manifest' };
  const manifest = promptRows.map(shapePrompt).filter((p): p is PromptSpec => p !== null);
  return { fleet, manifest, problem: null };
}

/** The dialable fleet: free rows only, brief order preserved. */
export function freeFleet(brief: TrialBrief): FreeModel[] {
  return brief.fleet.filter((m) => m.isFree);
}

/**
 * The catalog document. Built once from the brief and reused verbatim for
 * both `--list` output and the register body. The base URL is OpenRouter's
 * public endpoint on purpose: the gateway's trial address changes between a
 * live run and its replay, and this document must not.
 */
export function catalogFromBrief(brief: TrialBrief): CatalogDoc {
  const seen = new Set<string>();
  const models: CatalogDoc['providers'][number]['models'] = [];
  for (const m of brief.fleet) {
    if (!m.isFree || seen.has(m.dialId)) continue;
    seen.add(m.dialId);
    models.push({ id: m.dialId, free: true, context_length: m.windowTokens });
  }
  if (models.length === 0) return { providers: [] };
  return {
    providers: [
      { id: 'openrouter', base_url: PUBLIC_BASE, key_env: KEY_ENV_NAME, wire: 'openai', models },
    ],
  };
}

/** True when the model clears every requirement in the set. Unknown = unmet. */
export function modelQualifies(model: FreeModel, needs: NeedSet): boolean {
  if (needs.tools === true && model.canTools !== true) return false;
  if (needs.vision === true && model.canVision !== true) return false;
  if (needs.json === true && model.canJson !== true) return false;
  if (needs.minWindow !== undefined && (model.windowTokens === null || model.windowTokens < needs.minWindow)) {
    return false;
  }
  return true;
}
