// FiveTry Failover Server — candidate ranking, requirement sniffing, cooldowns.
//
// Routing policy, in one place:
//   * "auto" means: every free model that qualifies for the request, best
//     published rating first. Rating ties break on the larger context window,
//     then on the dial id, so the order is a pure function of the brief.
//   * A request that names a concrete catalog model pins that model first;
//     the ranked list still follows it as failover material.
//   * After a refusal the model cools down for a few *requests*. The clock
//     here is a logical request counter, never the wall clock: a replay sees
//     the same request sequence and the same gateway answers, so it must walk
//     the same candidate lists. Cooling models are skipped while any ready
//     model qualifies; if every qualifier is cooling, the full ranked list is
//     used rather than surrendering the prompt.

import { bagOf, modelQualifies, type Bag, type FreeModel, type NeedSet } from './brief.ts';

/** Hard cap on upstream dials for one client request. The name is the brand. */
export const MAX_TRIES = 5;

/** Cooldown lengths, counted in later completion requests, per refusal kind. */
const SIT_OUT_RATE_LIMIT = 3;
const SIT_OUT_SERVER_ERROR = 2;
const SIT_OUT_NETWORK = 1;

export class SitOutBook {
  private untilTick: Map<string, number> = new Map();

  /** Is this model sitting out at the given logical tick? */
  cooling(dialId: string, tick: number): boolean {
    const until = this.untilTick.get(dialId);
    return until !== undefined && tick < until;
  }

  /** Bench a model after a refusal; `status` 0 means the call never landed. */
  bench(dialId: string, tick: number, status: number): void {
    const span = status === 429 ? SIT_OUT_RATE_LIMIT : status >= 500 ? SIT_OUT_SERVER_ERROR : SIT_OUT_NETWORK;
    const prev = this.untilTick.get(dialId) ?? 0;
    this.untilTick.set(dialId, Math.max(prev, tick + span + 1));
  }
}

/** Best-first ordering: rating, then window size, then id for stability. */
function byMerit(a: FreeModel, b: FreeModel): number {
  if (b.scoreMilli !== a.scoreMilli) return b.scoreMilli - a.scoreMilli;
  const aw = a.windowTokens ?? 0;
  const bw = b.windowTokens ?? 0;
  if (bw !== aw) return bw - aw;
  return a.dialId < b.dialId ? -1 : a.dialId > b.dialId ? 1 : 0;
}

/**
 * The ordered candidate list for one request: qualifying free models, ready
 * ones first (best first), cooling ones only as a last-resort tail. A pinned
 * model (an explicit non-"auto" choice that qualifies) leads the list.
 */
export function candidateQueue(
  fleet: FreeModel[],
  needs: NeedSet,
  book: SitOutBook,
  tick: number,
  pinnedId: string | null,
): FreeModel[] {
  const qualified = fleet.filter((m) => m.isFree && modelQualifies(m, needs)).sort(byMerit);
  const ready = qualified.filter((m) => !book.cooling(m.dialId, tick));
  const queue = ready.length > 0 ? ready : qualified;
  if (pinnedId !== null) {
    const at = queue.findIndex((m) => m.dialId === pinnedId);
    if (at > 0) {
      const [pin] = queue.splice(at, 1);
      queue.unshift(pin);
    }
  }
  return queue;
}

/** Union of two requirement sets: a flag required by either side is required. */
export function mergeNeeds(a: NeedSet, b: NeedSet): NeedSet {
  const out: NeedSet = { ...a };
  if (b.tools === true) out.tools = true;
  if (b.vision === true) out.vision = true;
  if (b.json === true) out.json = true;
  if (b.minWindow !== undefined) out.minWindow = Math.max(out.minWindow ?? 0, b.minWindow);
  return out;
}

/** Does a message content array carry an image part, in either wire's shape? */
function contentHasImage(content: unknown): boolean {
  if (!Array.isArray(content)) return false;
  for (const part of content) {
    if (!bagOf(part)) continue;
    if (part.type === 'image' || part.type === 'image_url') return true;
    if (part.type === 'input_image') return true;
  }
  return false;
}

/** Requirements sniffed from an OpenAI-shaped request body. */
export function needsFromChatBody(body: Bag): NeedSet {
  const needs: NeedSet = {};
  if (Array.isArray(body.tools) && body.tools.length > 0) needs.tools = true;
  if (bagOf(body.response_format)) {
    const kind = body.response_format.type;
    if (kind === 'json_object' || kind === 'json_schema') needs.json = true;
  }
  if (Array.isArray(body.messages)) {
    for (const msg of body.messages) {
      if (bagOf(msg) && contentHasImage(msg.content)) needs.vision = true;
    }
  }
  return needs;
}

/** Requirements sniffed from an Anthropic-shaped request body. */
export function needsFromMessagesBody(body: Bag): NeedSet {
  const needs: NeedSet = {};
  if (Array.isArray(body.tools) && body.tools.length > 0) needs.tools = true;
  if (Array.isArray(body.messages)) {
    for (const msg of body.messages) {
      if (bagOf(msg) && contentHasImage(msg.content)) needs.vision = true;
    }
  }
  return needs;
}
