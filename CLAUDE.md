# CLAUDE.md — Hydra (Opus-Deipseek)

Single-file (`bot.py`) multi-model Discord bot. EVA/MAGI-themed heuristic router over
**Claude (Balthasar) · DeepSeek (Melchior) · Gemini (Caspar)** plus open-weight heads
**Qwen (Rei)** on Alibaba Model Studio (US-Virginia) · **GLM (Asuka)** on Fireworks, **Mistral (Mari)** on its own EU API, and
**Kimi K3 (Kaworu)** — Moonshot API by code default, but the live config.json overrides it onto
Fireworks serverless (`providers.kimi.backend="fireworks"`, rides `FIREWORKS_API_KEY`; override-only
in routing). Two-tier memory,
prompt caching, bookclub mode, Mandarin (`!speak`) + French (`!french`) TTS,
research panel (`!research`), simulator mode (`!dummy` — base-model transcript completion),
and cost + carbon tracking.

This file is the working status + remaining-work tracker for the "Hydra Restructure"
spec. The full spec lives outside the repo (Sarah's Claude.ai artifact); this file is
self-contained — you don't need it to pick up the remaining work below.

## Opus 5.5 migration — 2026-10-05

- Claude uses `claude-opus-5-5`, $4/M input, $20/M output, $0.20/M cache reads.
  Five-minute cache writes cost $5/M; one-hour writes cost $8/M.
- `!summarize` uses adaptive-low and a 4096-token budget, selects text blocks,
  and refuses to store empty, refused or truncated output. Other Claude effort
  choices are retained; signed tool continuations keep their original context.
- `ModelProvider.total_token_cost` persists token dollars rather than repricing
  lifetime counters after a model upgrade. Old Claude saves retain their former
  $5/$25/$0.50 estimate. Cache-write surcharges are included for new calls.
- Hydra's Alibaba US Qwen cache rate is confirmed at $0.014/M. ISAIC's Fireworks
  Qwen backend retains its separate rates.
- The sibling ISAIC and companion checkouts receive the corresponding migration.
  The older `E:\Documents\Stuff\isaic-slack-bot` checkout was removed at the user's
  request after verifying it was clean and its HEAD was in the retained checkout.
- FireRouter is a standalone, dry-run-first pilot in `scratchpad/eval_firerouter.py`;
  see `docs/firerouter-evaluation.md`. Production routing, personas and provider
  identities are unchanged. No live FireRouter evaluation has run.
- Offline migration regressions: `scratchpad/test_opus55.py`. The runner has
  separate no-network checks in `scratchpad/test_eval_firerouter.py`.
- Validation: migration/provider/panel/participation checks pass in Hydra and
  ISAIC; companion's full 67-test suite passes. Three synthetic live Opus 5.5
  checks passed (low, high, summary; $0.002104 total token cost). Windows startup
  validation below also confirmed Discord login; Slack remains offline-validated.

## Windows autostart — 2026-10-05

- Task `HydraOpusDeipseekAutostart` runs one minute after the user's Windows login,
  matching Ressapanda. It uses the interactive user's normal privileges, with no
  stored password, and permits battery operation.
- `tools/launch_bot.ps1` is the background launch entry point: correct `.venv` and
  working directory, hidden window, UTF-8/unbuffered logs under ignored `logs/`,
  a launcher mutex and a running-process check to avoid duplicate Discord clients.
- `tools/install_startup.ps1` installs/updates this checkout's task. The task is a
  logon launcher, not a crash-restarting service. `-CheckOnly` on the launcher
  validates prerequisites without starting the bot.
- Verified: task exit code 0, Discord login with Opus 5.5, and a second invocation
  kept the original Python parent/child pair without creating another log.

## Run / verify
- `python bot.py` — needs `.env` (`DISCORD_TOKEN` + ≥1 model key) and `config.json`
  (`allowed_channels`, `default_model`).
- Syntax gate (no deps): `python -c "import ast; ast.parse(open('bot.py',encoding='utf-8').read())"`
- Import/logic test: `python -c "import bot"` works (deps installed). On Windows console
  prefix `PYTHONIOENCODING=utf-8` when printing bot output (emoji/CJK/IPA).

## Editing rules (load-bearing — read before touching bot.py)
- **Graceful degradation:** a missing API key disables ONLY that provider. The gating key follows
  the *active backend* (`provider.api_key_env`): by default `FIREWORKS_API_KEY` gates GLM, **Qwen has
  its own `DASHSCOPE_API_KEY`** (Alibaba US-Virginia — region-locked key; Qwen's `fireworks` backend
  moves it back onto the Fireworks key), and **Mistral has its own `MISTRAL_API_KEY`** (`api.mistral.ai`) — its flagship (Large 3)
  is on-demand-only on Fireworks, not serverless. `self_hosted`/`vertex` backends have no key and aren't
  key-gated (operator owns the local server / GCP ADC). Absent keys disable exactly those providers.
- **Config-driven registry (Phase 0).** `ProviderRegistry.from_config()` (built once in `__init__`
  from `config.json`'s optional `providers` block) wires every provider's client + backend. The 6
  `ModelProvider(...)` constants are the code-side default spec (pricing/energy/quirks +
  `sdk_type`/`api_key_env`/`base_url`/`backend`/`backends`); config supplies operator overrides
  (`enabled`/`backend`/`model`/`routing_tags`) + the `platform` toggle. **No `providers` block ⇒
  byte-for-byte the old behavior.** `routing_tags` is present but INERT (routing stays in
  `_estimate_confidence`).
- **`provider.name` is the canonical routing key.** `_select_model`, `_estimate_confidence`,
  `panel_members`, cost, and persistence all branch on it. NEVER rename it. (`provider.id` — the
  lowercase config key — is separate; it keys `config.json` + `self.clients`.) Display names and
  command aliases are a separate theme layer on top (see naming below).
- **Dispatch** routes on `provider.sdk_type` (`anthropic` | `openai_compatible` | `gemini`), set by
  the registry. Claude → anthropic SDK; Gemini bookclub → native `cachedContents` (or Vertex
  `CachedContent` when `backend="vertex"`); everything else → the OpenAI-compatible shim
  (`_generate_openai_compatible_response`) via `self.clients[provider.id]`. Per-provider wiring lives
  in the registry, not `__init__`.
- **Prompt-cache discipline — the system prompt must stay byte-identical across turns.** Every head
  but Claude does AUTOMATIC prefix caching, and a prefix cache matches the LONGEST COMMON PREFIX, so
  anything that churns turn-to-turn must sit DOWNSTREAM of everything stable — a change near the front
  invalidates the cache for itself *and every token after it*, conversation history included.
  `_generate_response` therefore splits the prompt: `stable_system` (persona/server context/plurality)
  stays the cached prefix, and `volatile_context` (thread index — ages tick every MINUTE; memory —
  changes on every `[note:]`; the sparse-memory nudge — flips at the `<3` boundary) is appended to the
  newest user turn by `_append_volatile_context`. **Do NOT move those blocks back into the system
  message** (that was the bug fixed 2026-11-07). Claude is the exception — it caches ONLY what carries
  an explicit `cache_control`, so `_generate_claude_response(cache_system=True)` marks the stable
  prefix; the panel/judge pass `cache_system=False` because a one-shot prompt pays the cache-WRITE
  premium ($5/M) and is never read back. Guarded by `scratchpad/test_prompt_cache_order.py`.
- **Routing attribution rides a ContextVar, not a parameter.** `_SELECTION_MODE` (just after
  `GRID_GCO2_PER_KWH`) carries who chose the model for the turn being billed; `record_usage` reads
  it. It defaults to `"system"`, so a NEW utility call path needs no change — but a new **chat**
  path MUST `_SELECTION_MODE.set(...)`, or its turns bill as machinery. Anything reached from
  inside a chat turn that is not itself a model choice must `set` **and `reset`** (see
  `_panel_complete`), because a bare `set` leaks for the remainder of that Task. Guarded by
  `scratchpad/test_selection_attribution.py`.
- **Images are legal ONLY in user turns.** Anthropic, DeepSeek and the OpenAI content-part schema
  all reject an image outside a user message, and ONE such block 400s every subsequent turn in that
  channel until it scrolls out of history. `fetch_thread_history` roles by author and includes our
  own messages, so any image the BOT posts is a candidate — `_strip_assistant_images` (just before
  `_is_real_bot`) is the guard, and it must keep running at the end of history assembly. Do NOT
  rely on `_strip_images_from_messages`: that one only fires for `supports_vision=False` providers,
  which is exactly why enabling Deepseek vision exposed this. Guarded by
  `scratchpad/test_assistant_images.py`.
- **Anything printed to Discord must keep its emphasis markers balanced.** A bare `_` is an
  italics delimiter and pairs with the NEXT one anywhere in the message — lines later included
  — so an identifier like `AMORTIZE_TRAINING` opens a run that swallows everything up to the
  next `_` (live, it ate the omission note's opening marker and orphaned its closing one). Put any
  identifier in a code span, and NEVER truncate a message at a character position: drop whole
  lines, which are individually balanced, and re-close any code fence the trim landed inside.
  Guarded by the emphasis check in `scratchpad/test_cost_summary_format.py`.
- **Model slugs drift** — verify in the live libraries (Fireworks retires serverless models on
  ~2-week notice by email). Re-verified live 2026-09-21: DeepSeek `deepseek-flash` (V4.1 Flash),
  Alibaba `qwen3.8-flash` (live on `dashscope-us` 2026-09-21), Fireworks `glm-5p3` + the `qwen3p8-max` /
  `deepseek-v4p1-flash` backend toggles (under `accounts/fireworks/models/`). 2026-08-02: Claude
  `claude-opus-5`. **Re-verified against official docs 2026-11-07:** `deepseek-flash` is still the
  right slug but has silently been served by the **`DeepSeek-V4.1-Flash-1017`** snapshot since
  2026-10-17 (pricing unchanged; the "Flash refreshes behind the same slug" precedent held again) —
  552B MoE / 8B active in / 16B active out is confirmed verbatim, so our `est_wh` comment is right,
  and it has had **native vision since 2026-09-10** (see the `supports_vision=False` note on the
  constant — left off deliberately, it's a routing decision). Gemini `gemini-3.1-pro-preview` is
  **still preview with no retirement date**, so "no stable 3.1-pro" remains true — but the line has
  moved two generations past us: the flagship is now **`gemini-4-argon`** (1M ctx, $2/$10 until
  2026-11-30 then **$4/$20**, and ⚠️ **no explicit caching** — it would break the bookclub
  `cachedContents` path), `gemini-3.8-flash` is the current Flash, and `gemini-2.5-*` auto-routes to
  3.8/3.5-lite on 2026-12-01. Moonshot `kimi-k3` is still served and NOT deprecated, but
  **`kimi-k3.1-pro`** and `kimi-k3.1-pro-highspeed` now exist (1M ctx; K2 series retires late Dec
  2026). Our old note that "K3 takes only `reasoning_effort=max`" was **wrong** and is now corrected in
  code: the whole K3 series always thinks and takes top-level `reasoning_effort` of
  `low`|`high`|`max`, **default `max`**, so sending nothing on a plain turn billed max effort. Fixed
  via the `plain_reasoning_effort` tier (see the 2026-11-07 status entry). Mistral on its own API uses
  `mistral-large-latest` (→ Mistral Large 3, released
  2025-12, 675B/41B MoE, Apache-2.0; $0.50/$1.50 per Mtok reconfirmed 2026-08 — note Mistral's
  *frontier* is now Medium 3.5 / `mistral-medium-latest` at $1.50/$7.50, which the Large alias
  will never pick up; Large stays the deliberate cheap+green pick). ⚠️ Large 3 is
  on Fireworks only as **on-demand/dedicated** (`mistral-large-3-fp8`), NOT serverless — so the
  own-API route stays correct AND is now the cheap one, not just the green one.
- New-provider **pricing + energy constants are estimates** flagged `VERIFY` in comments —
  confirm against the Fireworks pricing page before trusting `!cost` $ figures.

## Naming theme — now a toggleable **theme layer** (`config.json` top-level `"theme"`)
Themes are a **display-only skin**: `Flavor`/`Theme`/`THEMES` (just after `SIM_PROVIDER`) map each
`provider.id` → themed display name + command aliases + a persona note. Read in
`ProviderRegistry.from_config` (mirrors the `platform` toggle), applied onto `provider.display_name`
and the `self.alias_to_flag` map built in `__init__`. **The canonical `provider.name`, the
`**[Claude]**` label, and `MODEL_LABEL_NAMES` never change** — a Judah/Gold-Head turn is still
labeled `[Claude]` on the wire. Canonical bare prefixes (`!claude`/`!deepseek`/…) + `!think` work in
every theme; the theme only *adds* its flavor aliases. Three sets ship (default **`eva`** ⇒
byte-for-byte the old behavior):
- **`eva` — EVA/MAGI** (default). Trinity Balthasar/Melchior/Caspar; pilots Mistral=Mari (`!mari`),
  Qwen=Rei (`!rei`), GLM=Asuka (`!asuka`); sim=Dummy Plug (`!dummy`).
- **`isaic` — ISAIC** ("International System of AI Coopertition"), the twelve tribes: Judah=Claude
  (`!judah`), Joseph=Gemini (`!joseph`), Zebulun=DeepSeek (`!zebulun`), Naphtali=Mistral
  (`!naphtali`), Benjamin=Qwen (`!benjamin`), Gad=GLM (`!gad`), Levi=sim (`!levi`). **Now in code**
  (was docs-only); it's also the Slack bot's default skin (Phase 4).
- **`nightvale` — the five heads of Hiram McDaniels** (+ residents): Gold=Claude (`!gold`),
  Blue=Gemini (`!blue`), Green=DeepSeek (`!green`), Violet=Mistral (`!violet`/`!purple`), Gray=Qwen
  (`!gray`/`!grey`); plus Carlos=GLM (`!carlos`) and the Faceless Old Woman=sim (`!faceless`).

---

## Status — 2026-11-08

### ✅ `!cost` split: per-provider detail is now `!costdetail` (2026-11-08)
Sarah, on the live 8-provider summary: *"can you check whether the italics are where they're
supposed to be for this message? also can we make a separate command for per-provider cost lol"*
— and then *"convert the 31 km figure to miles, and say gasoline instead of petrol. We're
americans. the other measurements can stay metric."*
- **The italics answer: wrong in that message, already fixed in the running bot.** The paste was
  timestamped **9:44**; the bot restarted at **9:51:09** onto the commit that fixed it (09:50:45),
  so the live head already emits the corrected form — confirmed by rendering the summary and
  counting delimiters. The mechanism, visible in the paste itself: `AMORTIZE_TRAINING`'s bare `_`
  opened an italic run that closed on **the omission note's opening `_`, two lines later**, so
  everything between them rendered italic, `AMORTIZETRAINING` lost its underscore, and the note's
  trailing `_` was left orphaned — the stray `...2,000-char limit)_` at the end.
- **The guard is now the general invariant, not a patch on one identifier.** New harness check:
  strip fenced blocks and inline code spans, then assert the `**` and `_` counts are both EVEN,
  over every shape these commands emit (2 / 8 / 20 heads, worst case, both detail forms). It
  immediately caught a second instance of the same class **in my own 2,000-char backstop**: a cut
  at character 1,987 can land between a `**` pair and bold everything after it. The backstop now
  drops whole **lines** (each line is internally balanced) and re-closes the code fence if the trim
  lands inside the ranking table.
- **`!costdetail` — aliases `!breakdown` / `!detail`, optionally one head.** Detail is the
  *precision* half of the answer and grows linearly with the fleet, while `!cost` is the *shape* of
  the answer and has to stay one scannable message — which is exactly why it kept being
  trimmed. `!cost` is now **1,283 chars at 8 heads** (it was 1,890 *with three providers' detail
  already dropped*) and ends in a `📋 Breakdown: !costdetail` pointer. `!costdetail` goes
  through `_send_response`'s chunking path, so **all 8 rows render in full** and nothing is ever
  declared omitted; it also spells "requests" out again and keeps the gate's full 24-char name,
  since width is no longer load-bearing there.
- **`!costdetail <head>` resolves through the router's own alias map**, so theme aliases work
  (`!costdetail rei` → Qwen), plus id / name / display name and an unambiguous name prefix —
  which is the only handle the **Haiku participation gate** has, since it bills real money and has
  no alias. A head that has never billed says so rather than rendering `$0.0000`.
- **Row building is shared.** New `_cost_rows()` builds the rows (in-flight Gemini cache storage
  included) for BOTH commands, so the rankings and the exact per-provider numbers can never
  disagree about what was billed; `mass()` became module-level `_mass_co2` for the same reason.
- **The two ranked lists are budgeted** (`fit_list`, 330 chars): they carry EVERY provider at full
  name length, so they grow fastest with the fleet. They keep whole entries and append
  `· +N more` instead of being cut mid-name. At 8 heads they cost ~150 chars, so this never
  bites in practice — it exists so a growing fleet degrades legibly.
- **US units for the car equivalent.** `CO2_G_PER_KM_PETROL_CAR` (250) →
  **`CO2_G_PER_MILE_GAS_CAR` (400)**, which is the EPA's own per-mile figure — so the constant
  is now the *source* value rather than a g/km conversion of it, one unit hop closer to the
  citation. Sub-mile masses read in **feet** (`0.4 miles` would invite precision this estimate has
  not got), one mile is singular, and "petrol" became "gasoline". **Only the car went imperial** —
  energy stays kWh, mass stays kg/g, grid intensity stays g/kWh.
- Offline-validated: `scratchpad/test_cost_summary_format.py` 65 → **85/85** (the emphasis
  invariant, including a non-tautology check that it really catches the original bug; the detail
  command's full 8-row output; single-head and never-billed branches; the budgeted lists; and every
  unit change) and `test_selection_attribution.py` 34 → **35**. Moved checks now read
  `get_cost_detail` rather than searching `!cost` for strings that left it.
- **PORTED TO ISAIC IN FULL** (same day, Sarah: *"we should probably honestly mirror the cost
  stuff to ISAIC as well"*). ISAIC had **none** of the stack — no ranked columns, no everyday
  equivalents, no per-Mtok or per-$, no attribution, no `cost_mode="plan"`; just the old
  registry-ordered list. All of it is there now, **extracted verbatim from `bot.py` by the port
  script rather than retyped**, so the two copies cannot drift in wording. Three adaptations were
  needed, and only three:
  - **A lone `~` is STRIKETHROUGH in Slack mrkdwn** (Discord needs `~~`), so every tilde in the
    rendered output becomes `≈` — which is why ISAIC already used `≈` everywhere.
    It is the AMORTIZE_TRAINING underscore bug with a different character: an odd count silently
    strikes out everything in between. The shared harness now checks it under
    `BOT_MODULE=isaic.core`. **This is the reason the summary cannot be shared byte-for-byte.**
  - **The char budget stays Discord's 1,990** even though Slack chunks rather than rejecting,
    because a chunk boundary inside the fenced ranking table would render the remainder as code
    — and one budget lets both repos run one harness. Verified end-to-end through the real
    `md_to_mrkdwn`: 1,282 chars → 1,266, **one** chunk, fence markers 2, zero `**` left over.
  - **`cost_mode="plan"` had to come too**, since the ported summary references it: the field,
    BOTH dollar gates, `_BACKEND_FIELD_MAP`, the config validation and the startup line. ISAIC
    had only `metered`/`local`.
  `**bold**` needed no adaptation — `md_to_mrkdwn` converts it on send, so both repos write
  the same Markdown. `!costdetail` uses `channel.send` (which converts AND chunks) instead of
  Discord's `_send_response`.
- **Three harnesses are now shared** (copied into `isaic-slack-bot/scratchpad/`, run with
  `BOT_MODULE=isaic.core`): `test_cost_summary_format.py` (**90/90** there — the five extra
  checks are the Slack tilde rule), `test_selection_attribution.py` (**35/35**, made bot-agnostic
  the same way `test_assistant_images.py` was) and the updated `test_prompt_cache_order.py`
  (**32/32** — the `_panel_complete` wrapper landed in ISAIC too, so its copy needed Hydra's
  two-function check). ISAIC's full suite: **13 harnesses green**.
- ⚠️ ISAIC's port is **offline-validated only** — that checkout has no live
  `config.json`, so the mrkdwn was verified by running the real converter, not by posting to Slack.
- ⚠️ **Owes one live check:** `!cost` then `!costdetail` (and `!costdetail haiku`) in
  Discord after a restart.

### ✅ First live 8-provider `!cost` — confirmed working, three defects fixed (2026-11-08)
The fix above worked: the channel answered normally, Deepseek even explained its own outage. The
`!cost` that followed **live-confirmed both new features** — `🎲 Chosen by: auto 35% · yours
65% · 63.8M unattributed` and `💵 Per $ ... Claude 60 · ... Deepseek 5.3k g` — and exposed
three things no offline fixture had, because **the real fleet is EIGHT providers, not seven**: the
seven heads plus the Haiku participation gate.
- **The ranking table came apart.** `provider.name` for the gate is `"Haiku participation gate"`,
  24 chars against a `{:<9}` field, so the row overflowed, pushed the right-hand column out, and
  welded a row number onto the previous share: `8 Haiku participation gate   <$0.01  <1%8 Mistral`.
  Fixed with a display-only `short()` that truncates to 8 chars + `…`; **the canonical
  `provider.name` is untouched** (it is the routing key) and the full name still appears in the
  prose rankings and Detail, where width is not load-bearing. Pinned by a check that every ranking
  row is the same width and that no row matches `%\d`.
- **Discord ate an underscore.** The amortized-training line printed `AMORTIZETRAINING=0`: a bare
  `_` is an italics delimiter, and this one paired with the marker on a *later* line, consuming
  both. Now a code span: `` `AMORTIZE_TRAINING=0` ``. **Mirrored to ISAIC**, where Slack mrkdwn has
  the same rule. General lesson, worth remembering for anything new that prints an identifier: an
  underscore that reaches Discord or Slack needs backticks.
- **Detail vanished wholesale** (`(per-provider detail omitted — ...)`) because 8 providers blow
  the 1,990 budget. The guard did its job, but all-or-nothing was the wrong design: Detail is
  ordered by spend, so the rows worth reading are at the top. It now **fills to the budget** and
  declares what it dropped. A first attempt reserved 60 characters for the closing note while the
  note was 78, producing a **2,002-char message — still rejected, by the very feature meant to
  prevent that** — so the trim now shrinks until it genuinely fits, with a hard cut as a last
  resort so an over-length `!cost` is impossible however the head grows. 20 providers with 24-char
  names render in 1,988.
- **Not a bug:** the live line showed no `machinery` bucket. Correct — the participation gate's
  lifetime tokens all predate the counter, and it had not billed since the restart, so there was
  nothing attributable to machinery yet. The gate DOES appear in the per-Mtok and per-$ rankings
  (80 g/Mtok, 66 g/$), which is itself useful: it is the cheapest head per dollar after Mistral and
  Gemini.
- Offline-validated: `scratchpad/test_cost_summary_format.py` 49 → **65/65**, now including the
  real 8-provider fleet with the gate's actual 24-char name, row-width equality, the
  `%\d` collision, canonical-name immutability, graceful Detail degradation with an accurate
  dropped count, a 20-provider pathological case, and the code span. Full suite green.

### 🔴→✅ LIVE OUTAGE: an image in an assistant turn 400'd every turn (2026-11-08)
At 08:26 both heads started failing on *every* message in a channel — on a plain "salam aleykum"
that contained no image at all:
```
[Deepseek] 400 — Image in assistant message is not supported
[Claude]   400 — messages.9.content: 'image' blocks are not permitted within assistant turns
```
**Root cause.** `fetch_thread_history` appends an image block for *any* image attachment it finds,
and roles each message by AUTHOR (`role = "assistant" if _is_real_bot(msg) else "user"`). Our own
messages are deliberately included in history (that is how the bot sees what it said), so **a bot
message carrying an image — one of our own LaTeX renders, a chart — becomes an illegal assistant
image block, and from then on EVERY turn in that channel 400s until it scrolls out of the window.**
`messages.9` was ours, not the user's.
- **Why it surfaced the morning after the vision flip.** `_strip_images_from_messages` drops image
  content wholesale for `supports_vision=False` providers — an **accidental** shield that happened
  to cover Deepseek. Enabling Deepseek vision (2026-11-07) removed it. **Claude never had that
  shield at all**, so this has been live for Claude for as long as the bot has posted images into a
  channel; it just needed an image followed by any message. Both heads failing at once is the tell
  that the poison was in shared history, not in one provider's dispatch.
- **Fix: one provider-agnostic invariant.** New module-level `_strip_assistant_images(messages)`
  rewrites any image block in a non-user turn to `[Image omitted: images are only valid in user
  turns]`, called at the END of `fetch_thread_history` — so whatever route an image takes in, it
  cannot leave. The attachment loop additionally skips the download for bot messages and appends
  `[Image posted: <filename>]`, which also saves a pointless base64 fetch and the tokens the block
  would have cost. The placeholder keeps the only part a model can use in an assistant turn: that
  an image was posted, and which file.
- **The constraint is universal, not Anthropic-specific.** Anthropic rejects it outright; DeepSeek
  400s (its own docs say images are valid in user messages only — recorded here yesterday); and the
  OpenAI content-part schema the shim targets carries `image_url` inside **user** content only. So
  the guard belongs in history assembly, not in any one dispatch path.
- `_strip_images_from_messages`' docstring was stale too ("Deepseek is currently the only text-only
  provider") — it now names the real text-only set (Mistral, Qwen, GLM, Kimi, sim) and carries a
  ⚠️ that it is **not** the assistant-turn guard.
- **Mirrored to ISAIC**, which had the identical latent bug: same builder, same role line, same
  vision flip.
- Offline-validated: new `scratchpad/test_assistant_images.py` (**21/21** in both repos) — the
  invariant, user turns left untouched, every content shape (string / text-only list / empty /
  None), `_msg_id` survival, no mutation of the caller's dicts, and an **end-to-end run of the real
  `fetch_thread_history`** over a fake channel reproducing the outage's exact shape: our bot posting
  `latex_render.png`, then a human saying hello. It also asserts the shim **would** still emit
  `image_url` on unsanitised input, so the check cannot rot into a tautology.
- ⚠️ **Owed:** the channel recovers as soon as the restarted bot rebuilds history, since history
  is re-fetched from Discord every turn and nothing poisoned is persisted — no state to clear.


### ✅ `!cost` gained a per-dollar ranking — the third ordering (2026-11-07)
Sarah: *"some kind of co2e/$ metric, where lower is better — if i'm going to spend a dollar on
tokens, what's the lowest waste i will make for it?"* It is the **dual** of the per-Mtok line, and
it ranks the fleet differently again:
**`💵 Per $ (lower better): Mistral 12 · Gemini 64 · Claude 77 · Kimi 108 · Qwen 660 ·
Deepseek 5.3k g`** — i.e. **a dollar spent on DeepSeek makes ~69× the emissions of a dollar spent
on Claude**, because a dollar buys ~29M DeepSeek tokens (at an 89% cache-hit rate) on a 550 g/kWh
grid, against a few hundred thousand Claude tokens on a 400 g/kWh one.
- **Per-Mtok and per-$ are duals, and which one is right depends on your binding constraint.**
  Per-Mtok is the metric when the WORK is fixed and you want the least emissions per unit of work;
  per-$ is the metric when the BUDGET is fixed and you want the least emissions per unit of spend.
  That is why they can rank in opposite directions without either being wrong: a cheap head on a
  dirty grid is simultaneously among the cleanest things to send a token to and the worst thing to
  spend a dollar on. Taken *alone* per-$ would say "buy the priciest model", which is only rational
  under a fixed budget — so it ships beside per-Mtok, never instead of it.
- **Realised, not rate-card.** Computed as `co2 / cost` from what actually happened, so cache hits
  and peak surcharges are already in it; a heavier hit rate makes a dollar buy *more* tokens and
  therefore *more* waste, which the harness pins directly (a cache-warm provider scores worse than
  an otherwise identical cold one).
- **Cache storage is excluded from the denominator** on purpose: those dollars bought no tokens, so
  counting them would flatter whichever provider leaks storage — exactly backwards.
- **Undefined, not zero**, where there is no per-token price: `cost_mode="local"`/`"plan"` rows are
  omitted from the ranking rather than rendered as perfectly clean. Sub-1 reads `<1`, four figures
  read `5.3k`.
- **Paid for in characters**, since the 7-head worst case had only ~60 left: the `Per Mtok` label
  lost its explanatory parenthetical, the order-of-magnitude caveat was trimmed, and Detail's
  "requests" became "req". Worst case is now **1,895/2,000** with Detail intact.
- Offline-validated: `scratchpad/test_cost_summary_format.py` 36 → **49/49** (the three-way
  ordering divergence driven on real figures, cache-warm vs cold, storage exclusion, local/plan
  omission, and both number formats).

### ✅ `!cost` now says WHO chose the model (2026-11-07)
Direct follow-up to the carbon correction below. The rankings are *volume × intensity* and say
nothing about whether anyone MEANT to send that volume — a head the router keeps picking is a
routing decision to revisit, while the same tokens chosen by hand are just what you wanted. New
line: **`🎲 Chosen by: auto 65% · yours 24% · machinery 11% · 61.5M unattributed`**.
- **Three buckets, named for who decided.** `auto` = the confidence heuristic's argmax, *and* the
  only-head-up case (nobody chose it, so crediting the user with a decision they never made would
  be wrong); `override` = a human pinned it — a bare `!claude`-style prefix, a `!prefer` channel
  pin, or the operator's `config.default_model`; `system` = not a model choice at all (panel
  members, the judge, bookclub recaps, `!summarize`, G2P, the participation classifier).
- **A ContextVar, NOT a threaded parameter** — which answers Sarah's earlier question about
  threading directly. `record_usage` has **11 call sites across five dispatch paths**, and the tool
  loop calls it several times inside ONE logical request, so threading would have touched all of
  them. `contextvars.ContextVar` is read inside `record_usage` instead, and asyncio gives each
  Discord event its own Task — a child Task gets a **copy** of the context — so it is exact under
  concurrent turns at zero churn at those sites. Default `"system"`, so every utility call bills as
  machinery without opting in. The one place that must force it back is **`_panel_complete`**,
  which is reached from inside a chat turn that has already set its own mode; it is now a thin
  `set`/`try`/`reset` wrapper around `_panel_complete_dispatch` (the old body — callers unchanged),
  and the `reset` is load-bearing: without it a `!research` would silently re-label the rest of its
  own chat turn as machinery.
- **Tokens, not requests.** `total_requests` is bumped at the call sites, where one logical request
  can record usage several times (tool loop, degraded retry), so a per-mode request counter would
  over-count. Tokens are also what both dollars and carbon follow. The figure is uncached + cached
  + output.
- **Legacy saves are reported, not guessed.** `tokens_by_selection` is absent from every save made
  before today, so the buckets sum to *less* than the token totals — the gap renders as
  `61.5M unattributed` instead of being folded into `auto`, shares are of the attributed subset,
  and the whole line is omitted while nothing is attributed. `load_stats` copies the dict rather
  than aliasing the saved one.
- **Char budget:** the line costs ~55–75 chars and the 7-head worst case is now **1,929** against
  Discord's 2,000. Paid for by dropping the duplicate everyday-equivalent from the
  amortized-training line (the carbon line above already anchors the reader). The 1,990 guard still
  catches the overflow case.
- Offline-validated: new `scratchpad/test_selection_attribution.py` (**34/34** — bucket mechanics,
  unknown-mode rejection, the tool-loop multi-call case, two interleaved asyncio turns keeping
  separate buckets, a child task not leaking back to its parent, the panel wrapper's reset, a
  persistence round trip incl. a pre-2026-11-07 save and a dict-aliasing check, all four
  `_select_model` branches driven through the real method, and the display's honesty about
  unattributed history). Full suite green; syntax + import gates pass.
- ✅ **Ported to ISAIC 2026-11-08** along with the rest of the cost stack (see above).
- ⚠️ **Owes one live check**: two real Discord turns — one plain (auto) and one `!claude`
  (override) — then `!cost`, to confirm the buckets fill as expected on the live wiring.

### ✅ Carbon constants corrected + a volume-free per-token ranking (2026-11-07)
Sarah pushed back on the headline from the `!cost` rewrite — *"i thought china had a lot of
wind/hydro/solar… and anthropic bought that colossus cluster lolsob"* — and was right on both
counts plus a third. The fix moves the ranking.
- **Claude `grid_gco2_per_kwh` 250 → 400.** The old 250 was justified in-comment as "AWS fleet:
  RECs + clean anchors", i.e. **market**-based accounting — but the field is defined two screens up
  as the intensity of the grid the endpoint *physically* runs on. Certificates don't move electrons,
  so the premium head was being flattered by our own definition's standards. Two facts make it
  worse: **2026-05-06 Anthropic leased ALL of Colossus 1** (Memphis TN, 300+ MW, 220k+ GPUs) from
  SpaceX — *leased*, not bought, and explicitly to serve Claude Pro/Max, i.e. **inference**, which
  is exactly what this field prices; and that site runs mostly on **~35 on-site simple-cycle methane
  turbines (~422 MW, against a 15-turbine permit — litigated by SELC/NAACP)**, topped up by
  ~150–250 MW of TVA grid. Simple-cycle gas is dirtier than the combined-cycle IPCC AR5 lifecycle
  median (~490 g/kWh); TVA's own 2025 system average is 721.69 lb/MWh = **~327 g/kWh**. Anthropic's
  AWS Trainium / GCP TPU capacity is cleaner and the blend isn't knowable from outside, so 400 = the
  US average = our global fallback. ⚠️ **Direction of residual error: if Memphis carries more of
  the serving load than the rest of the fleet, 400 is still LOW.**
- **China is no longer a coal-grid number.** Her instinct was right — the renewables build-out has
  pulled the national average to **~530 g/kWh** (MEE/NBS official, 2023), ~526 (Ember 2025), ~589
  (IEA 2025). Our 550 sits mid-spread and **stays**; what was wrong was the *contrast*. 250-vs-550
  implied a ~2× gap where the real one is **~1.3×**. Comment rewritten at the constant (east-CN
  runs above national average; Sichuan hydro ~112, Inner Mongolia coal much higher).
- **Effect on the headline:** on the same traffic DeepSeek's share of emissions falls 83% → 76% and
  Claude's rises 15% → 22%.
- **New: a volume-free `⚖️ Per Mtok` line.** Her third point was the sharpest — cumulative CO₂e is
  *volume × intensity*, and volume is a **routing/override artifact, not a property of a model**, so
  a cheap head that auto-routing picks constantly tops the CO₂e column while being among the
  cleanest things in the fleet per token. The new line is ordered cleanest-first and computed as
  `est_wh_per_1k_tokens × grid` (Wh per 1k tokens is numerically also kWh per Mtok, so the figure is
  just that product — no new state). On the live fleet: **Mistral 7 · Qwen 60 · Gemini 65 ·
  Deepseek 110 · GLM 180 · Kimi 200 · Claude 240 g/Mtok** — so **per token Claude is ~2.2×
  DeepSeek**, the exact inverse of the cumulative column. Both are true and they answer different
  questions ("what did we emit?" vs "what should we route to?"), which is why the volume-free one
  gets its own line instead of being left to inference. Guarded against regression by a check that
  the figure is unchanged at 10× the traffic.
  ⚠️ Those are the figures for the LIVE backends: Kimi reads 275 on the bare Moonshot
  constant (grid 550) but the live config pins `backend="fireworks"` (grid 400) → 200.
- Amortized training points the same way: ~8,000 tCO₂e (Claude, est.) vs ~1,000 (DeepSeek) over the
  same 100T-token horizon, so **counting training makes the premium head look worse, not better** —
  also as she guessed.
- **New: a 2,000-char guard.** These additions took the 7-head worst case to **1,908**, and a real
  summary carrying peak surcharges, tool-call clauses and a plan footer can pass Discord's hard
  limit — an over-length message is **rejected outright**, so `!cost` would have silently failed
  rather than truncated. Detail is now dropped (with a note) when the whole would exceed 1,990; the
  rankings and totals never are. Detail also picked up the `<1 g` treatment the rankings had.
- Offline-validated: `scratchpad/test_cost_summary_format.py` 23 → **36/36**. Full suite green.
- ⚠️ **Still estimates, and the remaining error moved.** The grids are now defensible; the squishy
  part is `est_wh_per_1k_tokens`, which is order-of-magnitude and unsourced per provider. **Gemini's
  130 is derived from Google's ~66% CFE figure, which is itself partly market-based** — the same
  critique as the old Claude number, left alone pending a location-based source. Never routed on.

### ✅ `!cost` rewritten: ranked by cost AND carbon, with everyday equivalents (2026-11-07)
The old layout listed providers in **registry order**, three lines each, which buried the one
genuinely surprising thing in the data: **cost and carbon rank differently.** On real figures Claude
is 59% of spend but 15% of emissions, while DeepSeek is 3% of spend and **83% of emissions** — cheap
head, dirty grid, far more tokens. A registry-ordered list hid that completely.
- **Two rankings side by side** in a fenced block (so Discord preserves the alignment), then the
  total, then the carbon line, then a per-provider **Detail** section. Insight first, precision after.
  Nothing from the old format was dropped: token counts, cache hit rates, Gemini cache storage, the
  `local` electricity-only label, the prepaid-plan footer, peak surcharge and per-request tool spend
  all survive in Detail, and the harness pins each of them.
- **Everyday carbon equivalents** on the total, because "7,199 g CO₂e" means nothing to most people:
  `_co2_equivalents()` renders km in an average petrol car (renamed to miles/gasoline 2026-11-08, see above; then `CO2_G_PER_KM_PETROL_CAR`, 250 g/km from
  the EPA's ~400 g/mile — **tailpipe only**, so it understates driving and is the conservative
  direction for a comparison that flatters us) and hours of PC gaming on a coal grid
  (`GAMING_PC_WATTS` 400 W × `COAL_GRID_GCO2_PER_KWH` 820, the IPCC AR5 lifecycle median for coal).
  Two anchors deliberately, not five — a longer list reads as precision the estimate doesn't have.
  All three are `.env`-overridable.
- Formatting judgements worth keeping: rankings use 2dp (`<$0.01` below that) so they stay scannable
  while Detail keeps 4dp; a sub-1% share renders `<1%` and a sub-gram mass `<1 g`, because `0%` and
  `0 g` read as *nothing* rather than *negligible*.
- Offline-validated: `scratchpad/test_cost_summary_format.py` (23/23 — the rankings really do come
  out in different orders, the equivalents appear and degrade to minutes/metres for small masses,
  every old field survives, and all seven heads render in **1,743 chars** against Discord's 2,000).

### ✅ web_search tool upgraded to the current variant (2026-11-07)
All three Claude `web_search` sites (`_generate_claude_response`'s default + both `_web_search` creates)
moved `web_search_20250305` → **`web_search_20260209`**, the current variant with dynamic filtering.
Supported on Opus 5.5/5/4.8/4.7/4.6, Sonnet 5.5/5 and Sonnet 4.6, so our `claude-opus-5-5` qualifies.
Two caveats recorded at the call site: dynamic filtering runs code execution server-side under the
hood, so a `code_execution` tool must NOT also be declared (two execution environments confuse the
model); and on Vertex only the basic variant exists — irrelevant here because Claude never runs on the
Vertex backend (that toggle is Gemini's). ⚠️ Worth knowing for the cache work above: **tools render
BEFORE `system`**, so the tool definition sits inside the `cache_system` breakpoint and is cached with
the prefix — the sibling grayson-bot measured `web_search`'s definition at **2,794 tokens**, which is
why the list must stay a byte-identical literal per call. A churning tool list would invalidate the
whole prefix. Owes the same live smoke test as the cache work.

### ✅ Kimi → K3.1 Pro, DeepSeek vision on, prepaid-plan cost mode (2026-11-07)
- **Kimi moved `kimi-k3` → `kimi-k3.1-pro`** (Moonshot's flagship, launched 2026-10-23). Both better
  AND cheaper, which is rare: **$2.00/$10.00 with $0.20 cached**, against K3's $3/$15/$0.30. Prices
  read off `platform.kimi.ai/docs/pricing/chat` **in a browser** — the table renders client-side, so a
  plain fetch returns an empty `<table>` and aggregators quote blended/reseller numbers. Context is
  exactly 1,048,576 for all K3 series. Confirmed Moonshot reports cache hits in the OpenAI-standard
  `prompt_tokens_details.cached_tokens`, so `_usage_cache_hits` already prices them right — the old
  "nonstandard key" worry is closed. Still override-only in routing. `kimi-k3` is NOT deprecated.
  ⚠️ **Inert on the live config**: `providers.kimi.backend="fireworks"` is set, and Fireworks serves
  only `kimi-k3` (plus kimi-k3-fast / -us routers) — there is no serverless `kimi-k3.1-pro`. Getting
  K3.1 Pro live means dropping that override and setting `MOONSHOT_API_KEY`, which moves Kaworu's
  egress back to the China-resident API and adds a second bill. Operator's call; the backend dict
  keeps its own correct K3 pricing meanwhile.
  ⚠️ **Known gap:** the K3 series bills cache WRITES separately by TTL ($2.00/MTok at the 5m default,
  $4.00/MTok at 1h) on top of reads. We model read rates only, so `!cost` under-reports a cold
  bookclub prefix on Kimi. Batch inference at 60% is also unmodelled (we never batch).
- **DeepSeek vision ENABLED** (`supports_vision=True`, Sarah's call). Native multimodal since
  2026-09-10: base64 data URL / http(s) URL / Files API id, JPEG·PNG·GIF·WebP, an upper bound of
  **1024 tokens per image**, ≤600 images and a 48 MiB body per request, images only in user messages.
  No new wiring needed — `_convert_messages_to_openai_format` already turns our Anthropic-style image
  blocks into OpenAI `image_url` data URLs for every shim provider. **Routing effect:** `vision_pool`
  in `_select_model` now includes Melchior, so an image message is no longer forced to Claude/Gemini —
  at $0.15/M vs $4/M that is the cheap eye, and the 1024-token image cap keeps looking cheap.
- **New `cost_mode="plan"` for prepaid subscriptions.** Answers "can we just keep per-token energy/CO₂
  for the monthly plans?" — yes, and it costs nothing in fidelity, because `get_energy_wh` /
  `get_co2_g` are computed from the token counters alone and never consult `cost_mode`. So a plan
  reports **byte-identical** energy, CO₂ and token counts to metered mode; only the per-token $ goes
  away, since usage draws down an allowance bought up front and charging per token would double-count
  against the flat fee. `plan_monthly_usd` carries the fee and surfaces as its own `!cost` footer line
  **below** the usage total (it is owed whether or not a token is sent, so folding it in would make a
  total that grows with traffic when it doesn't). Per-request tool spend still bills under a plan.
  Set it per provider in config: `providers.qwen = {"cost_mode": "plan", "plan_monthly_usd": 3}`;
  unknown modes, non-numeric fees and a fee without `cost_mode="plan"` all warn at startup instead of
  failing quietly. **Deliberately NOT built: a credits counter** — the OpenAI-compatible endpoint
  returns tokens, not Credits, and the token→Credit ratio is Alibaba's to define, so a local
  conversion would be invented precision. No provider ships as `"plan"`; absent config is byte-for-byte
  the old behavior. Offline-validated: `scratchpad/test_plan_cost_mode.py` (26/26, config scenarios in
  subprocesses since `from_config` mutates the module constants in place).
- **Gemini 3.1 Pro vs Gemini 4 Argon, priced out.** Argon is **cheaper than our current head right
  now and dearer from 2026-12-01**. At ≤200k: 3.1-pro-preview is $2/$12, Argon $2/$10 today → $4/$20 on
  Dec 1. Above 200k, 3.1-pro tiers to $4/$18 while Argon stays flat, so Argon is much cheaper today and
  roughly a wash after. For a 50k-in/2k-out turn: $0.124 on 3.1-pro, $0.120 on Argon now, **$0.240 on
  Argon from Dec 1** (~1.9×). The decider for us isn't the per-token rate though — **Argon does not
  support explicit caching**, so the bookclub `cachedContents` path (which takes a 452k fic from
  ~$1.81/turn inline to ~$0.45/turn cached) simply cannot be built on it. Staying on
  `gemini-3.1-pro-preview` is correct; it is still preview with no retirement date.

### ✅ Provider-doc sweep + per-request tool cost bucket (2026-11-07)
Triggered by unread provider mail. Every claim below was checked against the provider's own docs or
console, not inferred.
- **Rei is SAFE.** `qwen3.8-flash` is **not** on either Alibaba Model Studio retirement list — both
  notices were read in full (they render client-side, so a plain fetch returns an empty shell; the
  browser pane got them). Notice 2009 retires the OLD `qwen-flash` generation (`qwen-flash-us`,
  `qwen-flash-2025-07-28-us`, `qwen3-vl-flash-*-us`) and recommends `qwen3.6-flash-us` — our 3.8
  Flash is newer than the *replacement*. Notice 2000 (postponed to 2026-10-10) retires `qwen3-max`,
  `qwen3-vl-flash`, `qwen3-coder-plus`, the qwen3-* open weights, plus Alibaba-hosted
  `deepseek-v3.x`/`r1`, `glm-4.6`/`4.7`, `kimi-k2-*` — all third-party models we reach through other
  providers, so none of it touches us. Alibaba's own `deepseek-v4-flash`/`-us` are retired there too;
  irrelevant, Melchior is on DeepSeek's own API.
- **New: per-request tool cost bucket.** DeepSeek's Search & Fetch APIs bill **$0.75 per 1,000
  successful requests each, metered separately**, live since **2026-10-31 16:00 UTC** (a request bills
  only on HTTP 200 with a non-empty result; failures/empties/timeouts are free; `count` doesn't change
  the price). `ModelProvider` gained `search_cost_per_1k_requests` / `fetch_cost_per_1k_requests`
  (0.0 everywhere else, so no other head is affected), counters, and `record_tool_requests()` —
  deliberately NOT `record_usage`, because request-priced spend must stay out of the token buckets and
  the per-token energy/CO₂ estimate. Dollars are metered at record time like `total_token_cost`, so a
  rate change never reprices history; `get_cost()` adds it, and `!cost` gives it its own
  "(incl. $X for N search + N fetch tool calls)" clause. It bills even under `cost_mode="local"` —
  a self-host pays nothing for inference but Search/Fetch is a hosted service. **Nothing calls it
  yet** (search runs on Tavily); it exists so switching on native search can't silently under-report,
  the same failure shape as the 2026-09 cache-hit gap.
- **Stale constants found, deliberately left alone:** `DEEPSEEK_PROVIDER.supports_vision=False` is now
  factually wrong (native multimodal since 2026-09-10; `deepseek-flash` takes images as base64 /
  http(s) URL / Files API id, JPEG·PNG·GIF·WebP, ≤1024 tokens per image, 600 per request, user
  messages only). Flipping it changes routing — image turns would stop going to Claude/Gemini — so
  it's a product decision, documented at the constant. `_augment_with_url_extracts`' "Deepseek
  (text-only)" rationale is inaccurate for the same reason.
- **Fixed: Kimi plain turns no longer bill max effort.** The K3 series always thinks and defaults to
  `reasoning_effort="max"`, and our shim only sent the field under `!think` — so every plain `!kimi`
  chat turn ran at MAX reasoning on the priciest open head, silently, since July. New
  `plain_reasoning_effort` field (mirroring Claude's `CLAUDE_PLAIN_EFFORT`/`CLAUDE_THINKING_EFFORT`
  split); the shim now picks a tier per turn instead of only sending one under `!think`. Moonshot gets
  `plain="low"` / `think="max"`; **"low" is the floor, not "off"** — K3 cannot stop thinking.
  ⚠️ Both tiers are cleared on the **`fireworks` backend**, which is what the live config.json
  actually uses: "always max" is documented for Moonshot's hosted K3, Fireworks serves the open
  weights and may not implement the param, and a rejected param fails the whole turn — so guessing
  there would risk breaking plain `!kimi` to chase a saving. **On the live config this fix is
  therefore inert** until someone runs one `!kimi` and one `!think !kimi` against Fireworks; if the
  param is accepted, set the two tiers in the backend dict. Both tiers are now in
  `_BACKEND_FIELD_MAP` and — load-bearing — in the new **`_BACKEND_NULLABLE_FIELDS`**, because
  `_apply_backend` skips a `None` override otherwise and the backend would silently inherit "low".
  Offline-validated: `scratchpad/test_kimi_effort_tiers.py` (22/22, one file that resolves the
  Moonshot vs Fireworks routes by base_url so it runs unchanged in both repos despite their
  inverted defaults).
- Unverifiable from docs: DeepSeek's per-token price table and exact peak-hour window are published
  only as images, so our constants there stay as-is (off-peak = 50% of peak matches our 2× multiplier).
- Offline-validated: new `scratchpad/test_tool_request_cost.py` (31/31 — rate placement, separate
  metering, isolation from token/energy buckets, peak-surcharge exclusion, local mode, persistence
  round trip incl. pre-2026-11 saves, and no retroactive repricing).

### ✅ Prompt-cache ordering: volatile context moved off the cached prefix (2026-11-07)
Every head but Claude does automatic prefix caching, matched on the longest common prefix — and
`_generate_response` was joining three CHURNING blocks into the system message ahead of the history:
the thread index (per-thread ages are MINUTE-granular, so it changed every 60s while a thread was
fresh), the memory context (changes on every `[note:]`), and the sparse-memory nudge (appears/
disappears at the `<3` boundary). Because the system message is the head of the prompt
(`openai_messages.insert(0, ...)`), any tick invalidated the prefix cache for the *entire* prompt,
conversation history included — re-billing it at full input price. DeepSeek is $0.003/M cached vs
$0.15/M miss (50×), Kimi $0.30/M vs $3/M (10×), Qwen $0.014/M cached. The comment at the shim's
insert asserting "DeepSeek's server-side cache makes that ~free (~99% off)" was true only for the fic
block (always first) and false for everything downstream of the thread index.
- **Fix: the prompt is split.** `stable_system` (persona/server context/plurality — verified to
  contain no date) stays the cacheable prefix; `volatile_context` is appended to the newest user turn
  by the new `_append_volatile_context` helper, mirroring `_augment_with_url_extracts`, which already
  appends to the same turn for the same reason. One change at the assembly site fixes all five chat
  dispatch paths at once — no signatures threaded. Falls back to the old system-prompt placement when
  the newest turn isn't a user message (correct, just uncached, exactly as before).
- **The simulator deliberately keeps it in the preamble.** The tail of a sim prompt is transcript
  body, so a memory block there would render inside a `<speaker> body` line as something a user said.
  Nothing is lost — the sim rebuilds its whole transcript each turn, so there's no prefix cache to
  protect.
- **Claude now gets an explicit breakpoint.** Claude caches ONLY what carries `cache_control`, and
  plain turns had none (only bookclub did) — so the system prompt re-billed at $4/M instead of
  $0.20/M. New `cache_system` param (default True) marks the stable prefix, and extends the cached
  prefix over the framing on bookclub turns too. **The panel and judge pass `cache_system=False`**:
  their prompts are built per call and never read back, so a cache WRITE ($5/M at the 5-min TTL)
  would cost more than plain input.
- **Deliberately NOT changed:** the age strings stay minute-granular and the nudge keeps its `<3`
  boundary. Once the block sits at the tail, that churn is free, and minute precision is genuinely
  useful for recency — it was the PLACEMENT that was wrong, not the granularity. The `today` date in
  the `!research` judge prompt also stays: date granularity, single-shot, uncached, and load-bearing
  for the epistemics block. Invariant comments added at all three sites (thread-age builder, the shim
  insert, the assembly site) so this doesn't regress.
- Offline-validated: new `scratchpad/test_prompt_cache_order.py` (31/31 — helper shapes, the
  cross-turn stability invariant driven through the real `_generate_response`, tail delivery, the
  non-user-turn fallback, and all four Claude `cache_system`×reading-material combinations). Syntax +
  import gates pass; all 9 pre-existing harnesses still green (22/28/31/38/33/15/16 + 2 OK).
- ✅ **LIVE-VERIFIED 2026-11-07** via `scratchpad/live_cache_smoke.py` (3 real Opus 5.5 calls, $0.091;
  needs only `ANTHROPIC_API_KEY` — the cache claim is an API fact, not a Discord one, so this needs no
  running bot). Hydra's real cached prefix is **8,838 tokens** — far larger than grayson-bot's 4,416,
  because our system prompt carries seven personas, server context and the plurality block on top of
  the `web_search` definition.
  ```
  1. cold call                                read      0   uncached 8,912
  2. warm — thread age AND memory both changed read  8,760   uncached   157
  3. control: volatile back in the system prompt (the OLD shape)
                                              read      0   uncached 8,891
  ```
  Call 2 is the claim: the tail churned on every axis it has and the prefix still read back whole.
  Call 3 reproduces the pre-fix behavior deliberately and reads nothing, which is what EVERY turn did.
  **Measured saving:** the prefix went from 8,891 × $4/M = **$0.0356/turn** to 8,760 × $0.20/M + 157 ×
  $4/M = **$0.0024/turn**, i.e. **−93% of the prefix**; a 10-turn warm burst goes $0.356 → $0.066
  (**−82%**) including the one-off $0.044 write. Mirrored to isaic-slack-bot (same fix, its own smoke
  test still owed since it has no live `config.json`). ⚠️ Still unexercised: only the Discord wiring —
  `!cost` after two real channel turns.

### ✅ Provider-email sweep: dead Fireworks slugs, DeepSeek → V4.1 Flash + peak pricing, Qwen → Alibaba (2026-09-21)
Triggered by unread provider emails (Fireworks serverless deprecations 08-27 / 09-04 / 09-25,
DeepSeek V4.1 Flash + peak pricing, Azure subscription deletion). Every Fireworks/DeepSeek slug below
was verified with live calls (plain turn + web_search tool round-trip).
- **Qwen (Rei) was DOWN from 2026-09-04** — `qwen3p7-plus` 404s on Fireworks serverless. Fireworks'
  replacement `qwen3p8-max` is its only serverless Qwen and ~5× the price ($2/$6, cached $0.25).
  **Sarah's call:** the Discord bot's Qwen DEFAULT is now **Qwen 3.8 Flash on Alibaba Model Studio,
  US (Virginia)** — `backend="alibaba"`, `qwen3.8-flash`, `dashscope-us.aliyuncs.com/compatible-mode/v1`,
  `DASHSCOPE_API_KEY` (⚠️ region-locked: the key must be created in US-Virginia), $0.113/$0.382 flat to
  1M ctx — so Rei stays the cheap auto-routed coder (routing unchanged). Fireworks 3.8 Max stays as
  `backends["fireworks"]`. **isaic-slack-bot deliberately differs:** Fireworks 3.8 Max is its default,
  still auto-routed (Sarah's call). Only Alibaba serves 3.8 Flash (its open "Flash Next" weights are
  dedicated-GPU-only on Fireworks). `_BACKEND_FIELD_MAP` gained `max_context_tokens` +
  `est_wh_per_1k_tokens` so a backend carries its own model's context/energy. Thinks by default like
  the Fireworks Qwen (`enable_thinking` in extra_body toggles it; reasoning echo optional).
  ✅ **Key set + live-verified (2026-09-21/22):** no new Alibaba account was needed — grayson-bot
  already had a US-Virginia DashScope key (`DASHSCOPE_REGION=us-virginia` in its `.env`), and the
  same key works on `dashscope-us.aliyuncs.com` (no workspace id needed there, unlike grayson's
  `<workspace>.us-east-1.maas…` URL shape — both verified). Copied into this repo's git-ignored `.env`
  as `DASHSCOPE_API_KEY`; the registry now logs `🟢 Qwen enabled (model: qwen3.8-flash;
  backend=alibaba)`. Live checks on Rei's exact config: plain turn with thinking off, and a
  `web_search` tool round-trip with thinking left at its default (tool call → tool reply → final
  answer; `prompt_tokens_details.cached_tokens` reported, so `_usage_cache_hits` sees DashScope hits).
  Still owed: one real `!rei` in Discord once this bot is restarted wherever it runs (it wasn't
  running on the dev box). To fall back to Fireworks 3.8 Max: `providers.qwen.backend="fireworks"`.
- **GLM (Asuka)** `glm-5p2` retires 2026-09-25 → `glm-5p3` (same $1.40/$4.40; cached $0.14→$0.26;
  context 200k→1,048,576, so GLM can now take a bookclub text in full).
- **DeepSeek (Melchior) → V4.1 Flash** (`deepseek-flash`, Sarah's call). DeepSeek's own numbers: Flash
  wins every coding/agent bench (DeepSWE 74.2 vs 62.7, Terminal-Bench 3.0 30.0 vs 11.8, Codeforces 3471
  vs 3348); Pro keeps GPQA (92.4 vs 90.9) + HLE (42.7 vs 39.1). $0.15/$0.60/$0.003 off-peak. (The old
  $0.435/$0.87 constants were an expired promo — V4 Pro itself is now $0.66/$1.98.) Pro was slated to be
  force-routed to Flash on 09-14, then reprieved "until further notice"; V4.1 Pro is unreleased —
  re-evaluate when it lands. Verified live on `deepseek-flash`: thinking-disabled extra_body,
  thinking-on + reasoning_content echo, tool round-trip, `prompt_cache_hit_tokens`. `est_wh` 0.3→0.2
  (~8B in / ~16B out active). **discord-companion-bot migrated too** (`rules/llm_adapter.py`): the MTG
  strategist moved `deepseek-v4-pro` → `deepseek-flash` (keeps `reasoning_effort="medium"`, thinking on)
  and the actor's legacy `deepseek-v4-flash` alias → canonical `deepseek-flash` (thinking off, JSON) —
  both configs verified live. Its `!cost` got a new **exact-priced V4.1 Flash bucket**
  (`_deepseek_v41_cost`: $0.15 miss / $0.003 cache-hit / $0.60 out, 2× in peak hours, priced per call
  at record time; the adapter's `_Usage` now carries `prompt_cache_hit_tokens` as a subset of input).
  Every non-Pro DeepSeek name lands there; the old actor + pro buckets are frozen history at their old
  flat rates so lifetime totals aren't re-priced. The MTG cog that calls these factories lives outside
  that repo — re-check strategist memo length/compliance vs the May 23 V4-Pro baseline.
- **Peak-hour pricing implemented.** `ModelProvider.peak_windows_utc` + `peak_multiplier`;
  `record_usage` meters a peak request's extra cost into `total_peak_surcharge` (persisted as
  `peak_surcharge`; token buckets + energy untouched); `get_cost` = `_token_cost()` + surcharge; `!cost`
  shows "(incl. $X peak-hour surcharge)". DeepSeek: 2× during 01–04 + 06–10 UTC Mon–Fri (US Eastern:
  Sun–Thu 9pm–midnight + weekday 2–6am); Chinese public holidays are ignored (over-reports — safe
  side). Backends opt out with `peak_windows_utc: ()` (Fireworks is flat; self_hosted is local).
- **DeepSeek `fireworks` backend** slug was dead since 08-27 → `deepseek-v4p1-flash` ($0.30/$1.20/$0.006
  per Fireworks docs; its /models listing shows $0.22/$0.66 — VERIFY). Thinking-disable verified live.
- **Azure TTS is PARKED** (the free-credit subscription was deleted 2026-08-21, so `AZURE_TTS_KEY` is
  dead). New env off-switch **`AZURE_TTS_ENABLED=false`** (set in the live `.env`; unset/true = normal)
  blanks the key at startup, so every TTS path acts unconfigured: no Azure calls, inline
  `[[speak:]]`/`[[french:]]` markers collapse to plain text BEFORE any G2P LLM call, and `!speak` /
  `!french` reply "🔇 switched off". Mirrored to isaic. Offline: `scratchpad/test_tts_switch.py` (15/15 —
  flag parsing both ways on a real `ClaudeBot()` built in a temp dir, zero Azure/G2P calls when off).
  **To turn it back on:** pay-as-you-go subscription + an F0 Speech resource (0.5M chars/month free;
  S1 neural is $15/1M chars) → new `AZURE_TTS_KEY`/`AZURE_TTS_REGION` in `.env`, flip
  `AZURE_TTS_ENABLED=true`, restart.
- Mirrored to isaic-slack-bot (core.py constants + peak pricing, config.example.json → `deepseek-flash`,
  README). Offline: new `scratchpad/test_peak_pricing.py` (33/33 here incl. Qwen registry key-gating;
  29/29 in isaic) and every existing harness still green in both repos.
- ✅ **Cache-hit accounting gap — fixed (same day).** The shim used to read only DeepSeek's
  `prompt_cache_hit_tokens`, so Fireworks/Alibaba hits (reported in the OpenAI-standard
  `prompt_tokens_details.cached_tokens`) billed at full input in `!cost`. Module-level
  `_usage_cache_hits(usage)` now reads both (SDK object or dict; None-safe) — DeepSeek's field wins when
  non-zero so a response carrying both is never double-counted — at all 4 call sites (first call, tool
  loop, degraded retry, simulator). Qwen/GLM/Kimi cache rates are now live; plain `!gemini` shim turns
  also get the discount IF Google fills the standard field (unverified). Mirrored to isaic. Offline:
  `scratchpad/test_cache_hits.py` (22/22 in both repos — helper shapes, call-site migration, shim +
  simulator end-to-end over a fake client).

### ✅ Model refresh: Claude → Opus 5; every other head re-verified latest (2026-08-02)
`claude-opus-4-8` → **`claude-opus-5`** (same $5/$25 + cache rates — drop-in, `!cost` math
unchanged). ⚠️ The one behavior trap: **Opus 5 runs ADAPTIVE thinking when the `thinking` param is
omitted** (on ≤4.8 omitted meant off), and thinking tokens share `max_tokens`. Handling (Sarah's
call, same day): **plain turns run adaptive at a new LOW-effort tier** — `CLAUDE_PLAIN_EFFORT="low"`
+ `CLAUDE_PLAIN_MAX_TOKENS=8192` (headroom over the old 4096 cap, which thinking now shares) sent
explicitly by the `_generate_claude_response` else-branch and both `_web_search` creates. The
tiering keeps `!think` meaningful: plain chat / `!search` / bookclub recaps / G2P run adaptive-low;
`!think` + the auto-effort heuristic escalate to adaptive high|xhigh|max (16K/64K caps, unchanged).
**`!research` exception (Sarah's call):** Claude-as-panelist runs FULL thinking — adaptive-high +
its default web_search, the same profile as the judge — so the premium head digs rather than skims
(`_run_panel` passes `thinking=(provider is claude)`); non-Claude panelists keep thinking=False,
and the judge keeps its 2026-07-08 adaptive-high + web_search setting. The ONE call still
hard-disabled is
`!summarize` (200-token cap — adaptive there would blow the cap AND crash `content[0].text` on a
leading thinking block). Adaptive-on also removes the thinking-disabled tool-verbalization failure
mode, so no prompt mitigation was needed.
Non-Claude heads all re-verified CURRENT against official docs (see the slugs-drift bullet above):
DeepSeek V4-Pro (Flash was silently refreshed to build 0731 behind the same slug), Gemini
3.1-pro-preview (no stable 3.1-pro exists; no 3.2), Mistral Large 3 via `-latest`, Qwen 3.7 Plus,
GLM 5.2, Kimi K3. Fireworks pricing constants corrected (old VERIFY flags resolved — Qwen
$0.40/$1.60 cached $0.08, GLM cached $0.14) and the Kimi `fireworks` backend stub is now LIVE +
verified (see the Kimi entry). Watch: GLM-5.5 (Zhipu targets Aug 2026) and Qwen3.8-Max (WAIC
preview) are imminent but unreleased; DeepSeek peak-hour 2× surcharge announced, start TBA.
Mirrored to **isaic-slack-bot** (core.py + theme.py + config.example.json + README — same Opus 5
swap, adaptive-low plain tier, pricing fixes, **plus the full Kimi K3 port**; see the Kimi entry)
and **discord-companion-bot**
(`model_support` → `claude-opus-5` with `max_tokens_support` 4096→8192 for thinking headroom;
`model_default` already Sonnet 5, classifier already Haiku 4.5; fixed a latent `content[0].text`
crash in its `!summarize` — Sonnet 5 already defaults to adaptive thinking). The stale copy at
`E:\Documents\Stuff\isaic-slack-bot` (2026-06-28) was left untouched. Syntax + import gates pass.
⚠️ Owes live smokes: one plain `!claude`, one `!think !claude`, one `!search`, one `!summarize`,
plus the first live `!kimi` and `!think !kimi` on the Fireworks backend (verify the Fireworks shim
accepts `reasoning_effort` for kimi-k3 and reports cache hits in standard usage fields).

### ✅ Kimi K3 added as the 7th head (2026-07-17)
`KIMI_PROVIDER` (`id="kimi"`, name `Kimi` — canonical, never rename): Moonshot's **Kimi K3**
(released 2026-07-16 — 2.8T sparse MoE, 16-of-896 experts ≈ ~50B active, Kimi Delta Attention, 1M
ctx, largest open-weights model; weights themselves drop 2026-07-27). OpenAI-compatible on
`https://api.moonshot.ai/v1`, gated by **`MOONSHOT_API_KEY`** (graceful degradation — absent ⇒ only
Kimi disabled). Pricing $3/$15 per Mtok, automatic server-side cache hits at $0.30/M
(DeepSeek-style; ⚠️ VERIFY the usage-field name for hits on first live `!cost`). **Override-only in
routing** (`!kimi`/`!k3`, `!prefer kimi`) — it's the PREMIUM open head, above Claude on input, so
the argmax penalty is cost-safety, not just spec default. **Thinking quirk:** K3 takes only
`reasoning_effort="max"` and REJECTS the K2-era `extra_body {"thinking": ...}` — new provider field
`think_reasoning_effort` + a shim branch sends it when `!think` is on (⚠️ VERIFY on first live
`!think !kimi`). Themes: eva `!kaworu` (Fifth Child), isaic `!issachar` (the scholar tribe),
nightvale `!glowcloud`/`!allhail` (the Glow Cloud, ALL HAIL). In `panel_members_all` (`!research
all` is now up to 7 members when all keys are set). `backends` carries a **`fireworks` route that
went LIVE with the 2026-07-27 weights drop** — slug+pricing verified 2026-08-02
(`accounts/fireworks/models/kimi-k3` at $3/$15/$0.30-cached, matching Moonshot's own rates;
`routers/kimi-k3-us` is +10% for US-only/ZDR); the backend itself still owes a live smoke test.
**Ported to isaic-slack-bot 2026-08-02** with the lab-bot delta inverted: there the DEFAULT backend
is Fireworks serverless (rides FIREWORKS_API_KEY alongside Qwen/GLM; `moonshot` exists only as the
non-default parity toggle — China-resident API stays off the lab bot). The Discord bot ALSO now
runs Kimi via Fireworks, by operator override (`providers.kimi.backend="fireworks"` in the live
git-ignored config.json) — same $3/$15/$0.30 rates, US grid, one bill; MOONSHOT_API_KEY is no
longer required (delete the override to return to Moonshot). isaic port offline-validated:
`isaic-slack-bot/scratchpad/test_kimi_provider.py` (32/32 — adds a moonshot backend-toggle
round-trip scenario on top of the upstream 28). Upstream offline-validated:
`scratchpad/test_kimi_provider.py` (28/28 — constant shape, order, label strip, cost math, themes,
registry key-gating both ways, reasoning_effort shim branch). ⚠️ Owes live smoke tests: `!kimi`
round-trip, `!think !kimi`, cache-hit accounting in `!cost`, and a `!research all` with 7 members.

### ✅ `!load_text` accepts PDFs (2026-07-16)
Bookclub `!load_text` now takes `.pdf` attachments alongside `.txt`/`.html`/`.md`. `pypdf` is a new
optional dep (import-guarded like bs4 — missing ⇒ a pip-install hint, everything else unaffected; in
requirements.txt). Module-level `_extract_pdf_text(data) -> (text, page_count)` right after the
guard: opens owner-password-only "encrypted" PDFs with the empty user password, raises user-facing
`ValueError`s for truly locked PDFs and for scanned/image-only PDFs (no extractable text — OCR is
deliberately out of scope). Binary fetch via new manager `_fetch_file_bytes`; extraction runs in
`asyncio.to_thread` (CPU-bound). Thin-extraction heuristic (<100 chars/page avg) appends a ⚠️ to the
load summary instead of failing. Same `ReadingMaterial` path afterward, so chapter detection /
`!scope` / caching all apply. Offline-validated: `scratchpad/test_pdf_load.py` (16/16 — real
matplotlib-generated PDFs, both encryption shapes, blank pages, garbage bytes) + import gate with
pypdf blocked (`_HAS_PYPDF=False`, bot imports fine).

### ✅ `!research` panel: Gemini `thought_signature` 400 + DeepSeek tool-call dump (2026-07-10)
Two failures, both in the **shared tool loop** of `_generate_openai_compatible_response` — which
Gemini uses more than you'd think: `_panel_complete` routes everything non-Claude through the shim,
and `_generate_response` only picks the native Gemini path when reading material is loaded or
`backend="vertex"`. **Plain `!gemini` chat goes through the shim too.**
- **Gemini 400 "Function call is missing a thought_signature" (fixed).** Gemini 3 thinking models
  attach an encrypted per-call signature and reject the FOLLOW-UP request unless it's echoed back
  verbatim. Over the OpenAI shim it rides at `tool_calls[i].extra_content.google.thought_signature`;
  the loop rebuilt the assistant turn keeping only `{id,type,function}` and dropped it. So *any*
  Gemini turn that tool-called died on the second request — every `!research`, and every `!gemini`
  chat that decided to search. Bookclub was immune (native path). Now `_tool_call_extra_content()`
  reads it (typed attr → `model_dump()` → nested pydantic) and the rebuild echoes it back per call,
  never synthesizing one for a call that had none. ⚠️ Lowering `thinking_level`/`reasoning_effort`
  does NOT lift the requirement — don't "fix" it that way. Fallback if the shim ever returns no
  `extra_content`: stop attaching `web_search` to Gemini (it has native `google_search` grounding
  via `_search_for`). The outer `except` now prints that hint when it sees the string.
- **DeepSeek "no actual content (only unexecuted tool calls)" (fixed).** Live DeepSeek V4-Pro server
  bug (`deepseek-ai/DeepSeek-V3` issue #1244, open, ~2/19 completions): it serializes the call into
  `content` (`finish_reason="stop"`, `tool_calls=null`). Non-empty ⇒ sailed past
  `_is_provider_error`, counted as a surviving panelist, and reached the judge as an "answer". New
  `_looks_like_tool_call_dump()` catches the chat-template tokens (`<|tool▁…|>`, U+2581) / a tool
  name welded onto a JSON object / a whole-message tool-call object, deliberately high-precision so
  prose *about* `web_search` or JSON doesn't trip it. On detection (or on an empty completion) it
  retries once and, if still degraded, returns a real `Deepseek Error:` sentinel so the panel DROPS
  the member. The forced-text retry now **drops the `tools` array** instead of setting
  `tool_choice="none"` — a model that already decided to call a tool verbalizes the call when merely
  forbidden from emitting it. A blank with *no* tool intent keeps its tools (that retry is a re-roll).
- **Latent tool-loop bug also fixed:** only `web_search` calls got a `tool` reply, so any other tool
  name left a dangling `tool_call` id → malformed follow-up → 400. Every id now gets exactly one reply.
- **Panel/judge hardening:** `_run_panel` returns `(survivors, failures)` and logs the **full** error
  (the old `text[:120]` truncation is what hid this outage); the footer **names** failed members
  instead of `1 member(s) failed`; the judge's own output is now checked with `_is_provider_error`
  and a judge failure falls back to posting the raw panel answers instead of publishing
  `Claude Error 529: overloaded` as the synthesis; `_panel_complete` finally forwards `thinking=` on
  the openai path; the cost multiplier counts members *attempted*, not survivors.
- Offline-validated: `scratchpad/test_panel_fixes.py` (31/31 — true positives, false-positive guards,
  signature round-trip incl. `_strip_internal_keys` + JSON serialization); syntax + import gates pass.
  ⚠️ **Owes two live smoke tests:** (1) `!research` → Gemini survives a `web_search` round-trip;
  (2) grep the 7/9 logs for `⚠️ Deepseek returned empty content` — if that line is ABSENT, #1244 (not
  our own retry) was the cause, which is what the evidence says.

### ✅ Gemini bookclub cache — inline by default + real cost accounting (2026-07-08)
Root-caused a recurring Gemini prepay depletion (a persistent 452k-token bookclub cache). Fixes,
all offline-validated (`scratchpad/test_cache_fixes.py`, 20/20; import/syntax gates pass):
- **Inline is now the DEFAULT for Gemini bookclub.** `self.gemini_explicit_cache` (init in `__init__`
  from `providers.gemini.explicit_cache`, default **False**) gates BOTH `_ensure_gemini_cache` (native)
  and `_ensure_gemini_vertex_cache`. Off ⇒ the fic is inlined each turn (existing no-cache branch) and
  Google's **implicit** caching gives the read discount with **no per-hour storage bill** — no
  cachedContents to create/refresh/leak/`!uncache`. This CHANGES the old default (was explicit
  caching). Explicit stays fully intact behind the toggle. **`!explicitcache on|off`** (admin, persists
  via `_save_config`) flips it — `on` for a sustained back-to-back marathon (guaranteed discount, bills
  ~$0.45/hr storage for the 452k fic), `off` returns to inline AND drops any live cache. Startup prints
  the active mode.
- **Timezone bug fixed.** Cache expiry math mixed naive-UTC (`expireTime`, tzinfo stripped) with
  naive-LOCAL `datetime.now()` — on the UTC−4 host this made the code think a Google-expired cache was
  alive ~4h longer (dead-zone of full-price inline re-uploads; the sliding refresh also never fired).
  Now ONE aware-UTC clock: `_utcnow()` / `_as_utc()` everywhere in the cache path; `from_dict` coerces
  old (naive/mixed) persisted timestamps so nothing raises offset-naive-vs-aware.
- **`!cost` no longer under-reports storage.** Was a one-shot creation-time estimate (one 6h TTL,
  frozen) — a cache alive for days still showed ~$2.71. Now metered by REAL lifetime: `cache_created_at`
  per handle; live caches accrue on the fly in `get_cost_summary` (created→now, capped at expiry) and
  settle into `total_cache_storage_cost_est` at teardown (`_settle_gemini_storage`, idempotent, no
  double-count). `_gemini_storage_cost` is the shared helper.
- **`!uncache` no longer lies.** `_delete_gemini_cache` now returns success ONLY when actually gone
  (200/204/404, or a 403 whose body says "not found"); a bare billing/permission 403 → failure.
  `_drop_gemini_cache` deletes BEFORE clearing the handle and KEEPS it on failure (returns
  `(attempted, deleted)`), so a still-billing cache stays targetable instead of being orphaned.
  `!uncache` reports the real count and warns when a delete couldn't be confirmed.
- ⚠️ **Owes one live smoke test:** confirm Gemini implicit caching actually discounts a 452k prefix on
  `gemini-3.1-pro` (once the balance is topped up) — inline's whole premise. Worst case (no implicit
  hit) is ~$1.81/turn full input but still **no storage bleed**. Also verify `!explicitcache on` still
  creates/refreshes/deletes cleanly end-to-end.

### ✅ `!research` judge now verifies instead of dismissing post-cutoff facts (2026-07-08)
The Claude judge (`_judge`) was called with `claude_tools=[]` (web search **off** — "synthesise from
answers given") and a prompt telling it to "flag claims that look unsupported", with no cutoff
awareness. Result: real products released after its training cutoff (e.g. a June-2026 model) looked
"unsupported" → it declared them fabricated ("confirmed vapor") with false confidence. Fix:
- **Judge keeps web search** — dropped the `claude_tools=[]` (now defaults to the `web_search` tool),
  so it can VERIFY decision-critical / post-cutoff claims before ruling (selectively, not re-research).
- **Rewrote the judge system prompt** with an EPISTEMICS block: states today's date; "absence from
  memory is NOT evidence something is fake"; search to verify named products/versions/prices/dates/
  events that could postdate the cutoff; calibrated verdicts (CONFIRMED / CONTRADICTED / UNVERIFIED);
  NEVER say "fabricated/vapor" unless search *actively* contradicts it.
- **Judge now runs with `thinking=True`** (threaded through `_panel_complete`) — adjudicating conflicts
  + deciding what to verify is real reasoning, not a formatting pass. Other `_panel_complete` callers
  keep `thinking=False`.
- **Panel-member prompt** hardened against citation fabrication: URLs/quotes/dates/hashes may be cited
  ONLY from an actual `web_search` result in-conversation, never invented/reconstructed from memory; if
  it didn't search, attach no citations and flag the claim as from-memory. (DeepSeek *is* offered the
  Tavily `web_search` tool in the panel but chose not to call it, then fabricated URLs+hashes to satisfy
  "cite sources" — this is the source-side fix; the searching judge is the downstream catch.)
- **Empty-completion robustness** in `_generate_openai_compatible_response`: an empty `content` used to
  surface as a silent `''` → a dropped panel member with no retry (why a first `!research` could lose
  DeepSeek while the next worked). Now it logs `finish_reason` + `pending_tool_calls` and retries once
  (forcing `tool_choice="none"` if the 3-round tool cap was hit) — self-heals a transient blank and
  makes a persistent one diagnosable.
- Syntax/import gates pass. ⚠️ Owes a live smoke test (the exact failure query, once credits are up).

### ✅ Built (validated offline, needs live-key smoke tests)
- **Phase 7 — simulator mode, rung 1 (2026-06-27).** The **Dummy Plug** (`!dummy`/`!sim`): a base
  model continues the channel as an IRC/script transcript via `/completions` instead of chatting.
  `ModelProvider.completions_mode` + `sim_sampler` (§9.1/9.3); `_format_transcript` /
  `_parse_transcript_turn` / `_generate_simulator_response` (one dispatch branch ahead of the
  `sdk_type` cases); cost **and** carbon ride the same `record_usage` path as the chat heads (§9.2).
  Config-driven (`providers.sim.{enabled,base_url,model,api_key_env,sampler,search}`); **off by
  default** and **override-only** (excluded from the argmax; reachable via `!dummy` or `!prefer sim`
  — the spec's "designated channel"). Behavior-preserving on the default config (46-check offline
  harness + adversarial multi-agent review, findings folded in). **Scope (Sarah's call):** rung 1
  only — rungs 2–3 (ambient turn-taking, webhook personas; "the loom") were deliberately NOT built,
  and simulator mode is intentionally kept out of Phase 4 / lab planning. ⚠️ Owes one live smoke
  test against a real base endpoint.
  - **Query-driven search grounding (§9.2/§9.5, wired 2026-06-27).** A base model can't tool-call,
    so `_generate_simulator_response` now detects a search intent in the latest user turn
    (`_sim_search_query`: explicit `[search: …]` directive or a leading-question/lookup-cue
    heuristic) and **prepends** `_search_for(provider, query).text` to the transcript preamble —
    the sim analogue of the chat heads' tool-result append. **Off by default** (`sim_search`,
    flip via `providers.sim.search=true`) and gated on a real backend being available
    (`_search_backend_available`), so the default Dummy Plug never auto-searches. `SIM_PROVIDER`'s
    `search_backend` is back to `"tavily"`. Pasted-URL grounding still folds in regardless via
    `_augment_with_url_extracts`. Offline-validated (31-check `test_sim_search.py`).
- **Phases 0 + 2 + 3 — registry + Gemini Vertex + DeepSeek/Mistral backend toggles (2026-06-24).**
  Config-driven `ProviderRegistry` (`sdk_type` dispatch; `config.json` `providers`+`platform`),
  behavior-preserving on the default config (24-check offline harness passes). Gemini `vertex`
  backend + DeepSeek `fireworks`/`self_hosted` + Mistral `together`/`self_hosted` are code-complete
  with a `local` cost mode; the non-default backends ⚠️ owe live smoke tests. See the table +
  per-phase sections above.
- **Bookclub Gemini cache cost controls (2026-06-24).** Context-cache storage bills per token-hour
  for the whole TTL (~$11.6 for a 452k-token fic at the old fixed 24h), so: TTL is now a **6h
  sliding window** (`GEMINI_CACHE_TTL_HOURS` env knob; `_refresh_gemini_cache` PATCHes the expiry
  forward once past halfway → active discussion never expires mid-conversation, idle dies ~6h after
  last use), and **`!uncache`** drops just the Gemini cache while keeping the book loaded for all
  models (vs `!unload`, which removes the shared `ReadingMaterial` for everyone). Idle caches also
  self-expire, so day-to-day neither is needed.
- **Phase 1 — Qwen/GLM on Fireworks, Mistral on `api.mistral.ai`.** Qwen+GLM share one
  Fireworks key (cached input: 0.2× for Qwen, 0.1× for GLM — the old 0.5× assumption was wrong,
  corrected 2026-08-02); Mistral is on its own EU API (`MISTRAL_API_KEY`)
  because Mistral Large 3 isn't on Fireworks serverless — a happy accident: EU-resident + France
  ~nuclear grid (grid≈20 in `!cost`). Graceful degradation, EVA aliases, per-provider personas,
  in `!cost` and the `!research all` panel.
- **Routing — intentional deviation from the spec.** The spec said new heads "override-only."
  Per Sarah's call that cheap models should earn routing: **Qwen is in the auto-router as the
  cheap coder/mathematician** (wins routine code/math, hands complex/careful work back to
  Claude), **Mistral gets a narrow French-intent nudge**, **GLM stays override-only**.
  ⚠️ Do NOT "fix" this back to all-override-only — it's deliberate.
- **Beyond the spec (Sarah's additions):**
  - `!research all` — opt-in 6-model panel (plain `!research` stays the lean core trio).
  - **Energy/CO₂ in `!cost`** — per-provider Wh (`est_wh_per_1k_tokens`) × per-provider grid
    intensity (`grid_gco2_per_kwh`, follows the *endpoint* not the brand), plus a separate
    amortized-training line (`train_tco2e` over `MODEL_LIFETIME_TOKENS`). Env knobs:
    `GRID_GCO2_PER_KWH`, `MODEL_LIFETIME_TOKENS`, `AMORTIZE_TRAINING`. All flagged
    order-of-magnitude; never used for routing.
  - **`!french` tutor** — inverse of the Mandarin path: Azure `fr-FR-DeniseNeural`, Mistral-native
    G2P returning text+IPA+liaison note, IPA is display-only (no forced phonemes — Azure infers),
    inline `[[french:..]]` for models. Shares Azure config with `!speak`.
  - **Per-server system prompts (config.json `servers`, keyed by guild id) + runtime editors.** The
    system-prompt template has two placeholders filled per guild in `_build_system_prompt(...,
    guild_id=)`: `{server_context}` (the "context about this server" block) and `{plurality_section}`
    (the whole PluralKit block, dropped when a server sets `pluralkit: false`). Lookup order:
    `servers[<guild_id>]` → `server_default` → the `BotConfig` defaults. **The committed
    `default_server_context` is now GENERIC** — it tells the model it knows nothing about who's
    there and must not invent facts (secrets hygiene: no real names/servers in the repo; personal
    context lives only in git-ignored config.json or is set at runtime). `default_plurality_section`
    default is still `pluralkit=true`. The tidy-spacing `re.sub` runs whenever a guild deviates from
    the default. Config entries are validated/normalized by `_coerce_server_cfg` (non-dict → `{}` +
    warn; a quoted `"pluralkit"` → real bool + warn) so a config typo warns at startup instead of
    crashing `_build_system_prompt` on every message and silently muting the guild. `{server_context}`
    is substituted LAST so a custom context can't re-expand a placeholder token. `on_ready` prints
    each joined guild's name+id. **Runtime editors** (`!channels`, `!server_context`; admin/owner-only
    via `_can_edit_config`) mutate the in-memory state AND persist through `_save_config()`, which
    re-dumps `self._raw_config` to config.json (git-ignored) preserving all other keys — so a Discord
    edit and a hand-edit stay in sync. Rationale: the *old* prompt hardcoded "Sarah's projects /
    neuroscience / distributed databases" + an always-on PluralKit section, which made the bot invent
    plural-alter histories on servers that don't use PluralKit.
  - **Working-note extraction hardened.** All 5 generators + both web-search paths now call one
    `_extract_notes(text, guild_id)` helper (compiled `_NOTE_RE`, **case-insensitive** so
    `[Note:]`/`[NOTE:]` are caught, idempotent so it doubles as a safety net). Fixes a real leak:
    `_web_search` (Claude native) and the grounded-Gemini `!search` branch returned model text
    straight to Discord WITHOUT stripping `[note:]`, so a note tacked onto a `!search` answer leaked
    into the message.

### Remaining from the restructure spec
| Phase | Status | Priority |
|---|---|---|
| 0 — Provider registry refactor (`sdk_type`, config-driven) | ✅ **done** — `ProviderRegistry`, config `providers`+`platform`, `sdk_type` dispatch (behavior-preserving; offline-validated) | — |
| 1 — Mistral/Qwen/GLM on Fireworks | ✅ done (with routing deviation) | — |
| 2 — Gemini Developer-API default | ✅ already the default (operator: keep billing on) | — |
| 2 — Gemini **Vertex** backend | ✅ **code-complete** — `_generate_gemini_vertex_response` + `_ensure_gemini_vertex_cache` + backend-tagged caches; ⚠️ UNVERIFIED (needs GCP ADC) | Optional / lab-only |
| 3 — Provider backend toggles: DeepSeek (`api`/`fireworks`/`self_hosted`) **+ Mistral** (`api`/`together`/`self_hosted`) | ✅ **code-complete** — toggles in registry + `local` cost mode; `api` modes live, `fireworks`/`self_hosted`/`together` ⚠️ UNVERIFIED | Med — lab route |
| 4 — Slack platform abstraction | ❌ not built | **High — the big one** |
| 5 — Hermes delegation bridge | ❌ not built | Optional / lab-only |
| 6 — Ephemeral big-model co-op (405B base) | ❌ not built | Optional / lab-only |
| 7 — Simulator mode (§9, rung 1: transcript completion) | ✅ **code-complete** — Dummy Plug (`!dummy`/`!sim`), config-driven, override-only, cost+carbon tracked (offline-validated, 46-check harness); ⚠️ owes a live smoke test against a real `/completions` endpoint. Rungs 2–3 (ambient/personas) intentionally NOT built. | Done (fun) |

#### Phase 0 — Provider registry refactor — *✅ done (this session)*
`ProviderRegistry.from_config()` (right after the provider constants in `bot.py`) builds every
provider's client + backend from `config.json`. The 6 `ModelProvider(...)` constants stay as the
code-side default spec (pricing/energy/quirks + `sdk_type`/`api_key_env`/`base_url`/`backend`/
`backends`); config's optional `providers` block overrides `enabled`/`backend`/`model`/
`routing_tags`, and a top-level `platform` field is read+stored (Slack is Phase 4). **Hybrid by
design** — the richly-commented constants did NOT move into JSON; config is operator overrides only.
**No `providers` block ⇒ byte-for-byte the old behavior** (offline-validated: enabled set, dispatch,
cost, routing all unchanged). `__init__` is slimmed to bind registry results (`self.providers`,
`self.<id>_provider`, `self.clients`, `self.openai_compatible_clients`); dispatch switches on
`provider.sdk_type`; `pref_map`/`!prefer` are registry-built (so `!prefer qwen|glm|mistral` now work
— the one intentional additive change). `routing_tags` exists but is INERT. Package split deferred
to Phase 4.

#### Phase 2 (Vertex) — *✅ code-complete, ⚠️ unverified (this session)*
Developer API stays the default + no-train **as long as billing stays enabled on the Google project**
(operator setting — Sarah is paying). `gemini.backend = "vertex"` now routes to
`_generate_gemini_vertex_response()` + `_ensure_gemini_vertex_cache()` (lazy `vertexai` import;
`google-cloud-aiplatform` is an OPTIONAL, commented dep). **Cache handles are backend-tagged**
(`Gemini:developer_api` vs `Gemini:vertex` via `_gemini_cache_key()`) so a `cachedContents` id is
never reused against Vertex `CachedContent`; `_drop`/`_reconcile` handle both + the legacy untagged
key. ⚠️ **OWES A LIVE SMOKE TEST** — needs GCP ADC (`GOOGLE_APPLICATION_CREDENTIALS`) +
`GOOGLE_CLOUD_PROJECT`; the exact vertexai SDK surface (grounding ctor, `from_cached_content`) may
need a tweak per SDK version. `_reconcile_gemini_caches` still sweeps only developer-API caches
(Vertex orphans rely on the per-teardown drop — `TODO(vertex-reconcile)`).

#### Phase 3 — Provider backend toggles (DeepSeek + Mistral) — *✅ code-complete, ⚠️ unverified*
Both toggles are built in the registry (`backends` tables on the constants; selected via
`providers.<id>.backend`). `api` modes are live + unchanged; `fireworks`/`together`/`self_hosted`
are wired but ⚠️ **OWE LIVE SMOKE TESTS** (Fireworks/Together keys, a local vLLM). `self_hosted`
sets `cost_mode="local"` + `supports_server_cache=False` → `get_cost()` returns $0 and `!cost`
labels the line "local (electricity only)" while still showing the 🌱 energy/CO₂ (tokens counted on
the self-host grid). DeepSeek still defaults to its China `api` for the Discord bot (locked
deviation). The per-backend reference below still applies (values flagged `VERIFY` are estimates):

**DeepSeek** stays on its China `api` for the Discord bot (Sarah's call — cheap, and "the Chinese can
have the shitposts"). To add the toggle: `providers.deepseek.backend = api | fireworks | self_hosted`.
- `fireworks` → `accounts/fireworks/models/deepseek-v4p1-flash` (V4 Pro left Fireworks serverless
  2026-08; see the 2026-09-21 entry), shares `FIREWORKS_API_KEY`, flat Fireworks rates. Update its
  `grid_gco2_per_kwh` to ~400 (US) when on this backend.
- `self_hosted` → local vLLM/Ollama base_url; set `supports_server_cache=False`; add a `local`
  cost mode in `_record_*_usage()` (skip per-token $, label the turn `local`); V4-Flash or 4-bit
  for single-GPU.

**Mistral** is hardcoded to `api.mistral.ai` (set up this way because Mistral Large 3 isn't on
Fireworks *serverless* — `mistral-large-3-fp8` is on-demand/dedicated only). Same
toggle shape: `providers.mistral.backend = api | together | self_hosted`.
- `api` (current) → `api.mistral.ai`, `MISTRAL_API_KEY`, model `mistral-large-latest` (→ Large 3,
  $0.50/$1.50 per Mtok as of 2026-06). EU-resident + France ~nuclear grid (`grid_gco2_per_kwh=20`).
  **Best default for the Discord bot — keep it** (now the cheap route too, ~4× under Large 2).
- `together` → Together AI serverless (US) *does* host Mistral Large 3 — use for US residency / one
  fewer bill; set `grid_gco2_per_kwh≈400` + Together's rates. (Fireworks only has it on-demand —
  i.e. rent a dedicated GPU — not worth it for a chat bot.)
- `self_hosted` → Apache-2.0 weights, but Large 3 is 675B (multi-GPU). For single-GPU, self-host a
  smaller Mistral (Ministral 3B/8B/14B, Magistral Small 24B); reuse DeepSeek's `local` cost mode.

#### Phase 4 — Slack platform abstraction — *TODO, the big remaining chunk*
Introduce a `ChatPlatform` protocol; make `ClaudeBot`/`ConversationManager` depend on it instead
of `discord.py` directly. Extract `DiscordAdapter` (wrap existing logic, don't rewrite behavior);
build `SlackAdapter` on `slack_bolt` in **Socket Mode** (no public URL). Specifics:
- Threads → `thread_ts`; 👍/👎 calibration → `reaction_added`/`reaction_removed`; files →
  `files_upload_v2`; `!claude`-style commands → slash commands or `app_mention` parse.
- **Re-key persistence** `memories.json` on `(platform, team/guild, channel)` + a migration that
  namespaces existing Discord keys under `discord:<guild>` (else Discord/Slack clobber each other).
- **Formatting:** Slack is mrkdwn + Block Kit, not Markdown — audit every emit (code fences, bold,
  links, `!help` tables). LaTeX→PNG and the TTS MP3 paths are fine (image/file upload).
- **De-dupe** Slack's event retries on `event_id`/`client_msg_id`.
- Ship a Slack app manifest (scopes + event subs + slash commands). Tokens: `SLACK_BOT_TOKEN`
  (`xoxb-`), `SLACK_APP_TOKEN` (`xapp-`).
- This is where the **ISAIC naming skin** activates (Slack = lab bot).

#### Phase 5 — Hermes delegation bridge — *optional, lab-only*
Co-located Hermes Agent (separate service, thin bridge — do NOT merge codebases). Route `!research`
or a new `/task` to a local Hermes instance, stream the trajectory back via existing `send_text`/
`send_file`. **Egress discipline is a security rule, not a convenience:** force Hermes onto the same
hardened endpoints (Fireworks US/ZDR, self-hosted DeepSeek, Vertex); gate genuinely patentable
prompts to self-hosted only. Optional cron delivery into the lab channel. Hydra must still run
fully with Hermes absent.
**2026-07 framework update** (full revision in spec §10): **Hermes Cloud** (Nous-hosted agents on
Nous Portal, 2026-07-08) is explicitly NOT the lab route — it moves the agent runtime (memory,
trajectories, prompts) onto Nous infra, and default onboarding now funnels to Portal (OAuth +
Portal inference + Tool Gateway). Phase 5 = self-hosted Hermes with hand-set provider config;
treat Portal/Cloud opt-ins as egress surface. **v0.18.0** (2026-07-01) meanwhile strengthens the
bridge: Vertex is now a first-class Hermes provider (ADC OAuth, no static key — same GCP prereq
as Hydra's Phase 2 backend), so all three hardened endpoints are natively supported; cron gained
native Slack (Block Kit) + Discord delivery (seam 2 is now mostly configuration) and blocks
`base_url` overrides; the `llm.oneshot` gateway RPC is a concrete seam-1 hand-off surface. MoA
virtual models (panel→aggregator ≈ `!research`+judge) and `/learn` (skills from directories/URLs)
are useful but token-hungry / egress-sensitive — the hardened-endpoint rule extends to whatever
they read. **v0.19.0 "Quicksilver"** (2026-07-20; full entry in spec §10) strengthens the bridge
further: Fireworks is now a first-class Hermes provider (the US/ZDR default no longer rides the
generic OpenAI-compatible path); providers can be hard-disabled (`enabled: false` /
`excluded_providers` — turn Portal/OpenRouter off outright); cron delivery is durable
(delivery-obligation ledger survives gateway crashes); delegation gained unified concurrency caps
(`max_async_children` deprecated); MoA got cost caps (`reference_max_tokens`, per-turn fanout
cadence). ⚠️ New caveat: an `api_content` sidecar persists exact API bytes on local disk — disk
encryption/retention on the Hermes host joins the egress surface. The Portal/Cloud funnel
meanwhile deepened (desktop Cloud connection mode, in-terminal billing, Portal usage tags on
aux/MoA/delegate calls), confirming the self-host ruling; v0.19.1 (2026-07-30) is a bug-fix wave.
Spec §10.5 (new) collects principles borrowed from Anima Labs' Connectome for Phases
4/7 + memory; Connectome itself is research-grade, not a dependency.

#### Phase 7 — Simulator mode (§9, rung 1) — *✅ code-complete, ⚠️ unverified (2026-06-27)*
A second **generation mode** behind the same dispatch (NOT a second bot): `completions_mode=True`
routes a provider to `_generate_simulator_response()` (transcript completion) instead of the chat
path. The whole feature is three methods (§9.3): `_format_transcript` renders channel history as a
`<speaker> body` IRC log (user turns carry `Name:`, bot turns `[Model]`; a leading `[replying to …]`
block is dropped at the block level so embedded `]`/newlines can't corrupt the speaker line) + a
dangling `<Dummy>` continuation cue; `client.completions.create` continues it; `_parse_transcript_turn`
cuts back to one line (server `stop` + a client-side `<name>`-only cut — deliberately NOT `[name]`,
which would eat code/citation lines). `SIM_PROVIDER` ("Dummy"/"Dummy Plug", `id="sim"`) ships
`enabled=False` + `api_key_env=""` (self-hosted, keyless), so the registry leaves it OFF unless
config opts in — the one place `config_enabled` now defaults to the constant's `enabled` instead of
`True` (a no-op for the 6 always-on heads). **Override-only** is enforced in `_select_model`
(`auto_pool` drops `completions_mode`), with one deliberate exception documented in code: if the
base model is the *only* enabled provider (a sim-only box), plain messages route to it via the
`len(enabled)==1` shortcut. Sampler knobs (§9.3) split standard params (on the call) vs
`top_k`/`min_p`/`top_a`/`repetition_penalty` (in `extra_body`); unknown `providers.sim.sampler` keys
warn at startup. ⚠️ **OWES A LIVE SMOKE TEST** — stand up any `/completions` base model (a small
GGUF/Ministral base on vLLM or llama.cpp suffices — no 405B), set
`providers.sim = {enabled:true, base_url, model}`, then `!dummy <prompt>`; confirm the `[Dummy]`
turn lands in `!cost` with tokens + 🌱. **Query-driven search grounding** (§9.2/§9.5) is now wired
behind `providers.sim.search=true` (off by default): `_sim_search_query` detects a `[search: …]`
directive or a leading-question/lookup cue and `_generate_simulator_response` prepends the
`_search_for` snippets to the preamble (gated on `_search_backend_available` so no sentinel leaks).
Rungs 2–3 (ambient turn-taking, webhook personas) and any Phase 4 / lab-adjacent simulator work are
intentionally out of scope.

### Cross-cutting follow-ups
- ~~Verify **pricing**~~ — done 2026-08-02: Fireworks `qwen3p7-plus` ($0.40/$1.60, cached $0.08) /
  `glm-5p2` ($1.40/$4.40, cached $0.14) / `kimi-k3` ($3/$15, cached $0.30) and Mistral
  `mistral-large-latest` ($0.50/$1.50) all confirmed against official pages; constants updated.
  DeepSeek's peak-hour 2× pricing went live 2026-09-10 and is now metered (see the 2026-09-21
  entry). The **energy** constants (`est_wh_per_1k_tokens`, `train_tco2e`) are
  order-of-magnitude (Mistral `train_tco2e` is still the Large-2 LCA — Large-3's isn't published).
- Live smoke tests still owed: `!mari`/`!rei`/`!asuka` round-trips, `!french bonjour` (Azure
  fr-FR synth) and `!french how do you say …` (Mistral G2P), inline `[[french:..]]`. **Plus the
  new backends (Phase 2/3):** Gemini `vertex` (GCP ADC + `google-cloud-aiplatform`); DeepSeek
  `fireworks` (`!deepseek`) + `self_hosted` (local vLLM → shows `local` in `!cost`); Mistral
  `together` (Together key) + `self_hosted`. Verify the Together Mistral-Large-3 slug + Fireworks
  DeepSeek/Together pricing (flagged `VERIFY`).
- **Phase 7 simulator:** `!dummy <prompt>` against a real `/completions` base endpoint (transcript
  continuation, tokens + carbon in `!cost`); `SIM_PROVIDER` pricing/energy are `VERIFY` placeholders.
  Also smoke-test search grounding: `providers.sim.search=true` + a Tavily key, then a `!dummy`
  turn whose latest line is a question / `[search: …]` — confirm the snippets land in the preamble.
- Operator: keep Fireworks prepaid balance topped up (auto-reload) and Gemini billing current.
