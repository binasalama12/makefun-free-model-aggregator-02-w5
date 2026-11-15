# FiveTry Failover Server

A zero-dependency Node 22 server that puts the whole free tier of OpenRouter
behind one endpoint and speaks both client APIs: OpenAI Chat Completions and
Anthropic Messages, in plain JSON and in streaming (SSE) form.

Point any OpenAI- or Anthropic-shaped client at it, ask for model `auto`, and
it answers with a real completion from the best free model that fits the
request — picked from the catalog in the brief it was started with, ranked by
the published quality rating.

## Why "FiveTry"

Free models refuse. They rate-limit, they 5xx, they vanish mid-trial. This
server treats a refusal as routing information, not as an answer:

- Each request gets up to **five dials**. A 429, a 5xx or a network failure
  benches that model and moves straight to the next qualifying model.
- Benched models **sit out** the next few requests (longest after a 429),
  counted on a logical request counter — never the wall clock — so behaviour
  is deterministic and replay-safe. While any ready model qualifies, cooling
  models are not dialled at all.
- Every refusal along the way is reported on the final response in the
  `x-makefun-upstream-failures` header (a JSON array of
  `{"status","code","model"}` rows), so callers can see what the answer cost.
- A hard 4xx (not 429) stops the walk immediately: another model cannot fix
  a malformed request, and the dial is not wasted.

## Routing

- Candidates are the brief's free models that meet the request's needs:
  declared tools require a tools-capable model, a JSON `response_format`
  requires JSON mode, image parts require vision. The trial manifest's
  requirements and the request body's own signals are merged.
- Order is rating first (highest `rating_milli`), then larger context
  window, then dial id — a pure function of the brief.
- A request naming a concrete catalog model pins that model first, with the
  ranked list behind it as failover material.
- Every answer and every streamed chunk carries the id of the model that was
  actually dialled in its `model` field.

## Layout

- `main.ts` — process bootstrap: brief load, `--list`, registration with the
  trial env, HTTP routing, the five-try dial loop, and the 100 ms work poll
  that exits 0 when the env says the trial is done.
- `lib/brief.ts` — finds the model table and prompt manifest inside the
  brief file and builds the catalog document from them (nothing is
  hardcoded; `--list` and registration share the one built document).
- `lib/router.ts` — candidate ranking, requirement sniffing from request
  bodies, and the sit-out (cooldown) book.
- `lib/translate.ts` — Anthropic ↔ OpenAI translation for requests, answers
  and both streaming shapes, including tool calls and usage.

## Running

```sh
MAKEFUN_PROMPT_FILE=/input/prompt.txt node main.ts --list   # print the catalog
node main.ts                                                # serve (trial env vars set)
```

The server listens on `0.0.0.0` at an OS-assigned port (never 8080),
registers `{protocol_version, port, catalog}` with `$MAKEFUN_ENV_URL`, dials
models through `$OPENAI_BASE_URL/chat/completions` with the trial key in
`$OPENAI_API_KEY`, and polls `$MAKEFUN_ENV_URL/work` every 100 ms, exiting 0
as soon as it reports done.
