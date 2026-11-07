# FireRouter evaluation for Hydra and its forks

Status: prepared on 2026-10-05; no live FireRouter run or production routing change.

The question is whether FireRouter improves these bots' cost **at acceptable quality**, compared with their existing multi-model routing. An Opus-only comparison cannot answer that: Hydra already sends inexpensive work to Qwen and DeepSeek. Companion's emotional-support routing and ISAIC's provider/residency constraints require separate decisions.

## Stage 1: synthetic compatibility pilot

Run `python scratchpad/eval_firerouter.py` from Hydra for an offline preview. The script imports no bot, reads no memories, and changes no config. Eight cases cover casual conversation, code, math, Chinese, French, source-grounded bookclub discussion, a deterministic search tool, and emotional support. The last two use synthetic sources and dialogue. Each arm owns its own append-only multi-turn history. Bookclub provides a cacheable prefix, with a follow-up turn to exercise warm-cache behavior.

Compare direct `claude-opus-5-5` with `firerouter/opus`, both through the Anthropic Messages protocol and adaptive-low. Run a small billable pilot first:

```powershell
python scratchpad/eval_firerouter.py --live --env-file .env --cases code,bookclub --provider-key-configured
```

`--provider-key-configured` means you have already connected an Anthropic Provider Key in Fireworks. If you instead want the script to send your existing Anthropic API key to Fireworks on each call, explicitly use `--forward-anthropic-key`. It forwards only to `https://api.fireworks.ai/inference`. This is additional credential sharing; the offline preview does neither. No subscription credit is involved in this API-key experiment.

Fireworks documents that a route without closed-provider credentials can silently use open models only. The runner first forces the primary route with `x-routing-preference: 1` and requires a response model of `claude-opus-5-5`. This one probe is billable too. It then records the actual response model, raw usage counters, completed/partial responses, request IDs, tools, and whole-call latency. Retries are off, output caps are 2,048 tokens by default, and tool loops stop after four calls. There is no guaranteed dollar cap: check both balances and begin with the two-case pilot. [API setup and model attribution](https://docs.fireworks.ai/nexus/firerouter/setup)

Results go into a new ignored `scratchpad/firerouter-results/run-*` directory. Grade `blind-review.csv` before consulting the arm mapping in `results.json`. Score each session 0–4 against its rubric and separately flag critical failures. Empty outputs, API errors, invented citations, ignored requests, truncated responses, or incorrect tool arguments count against the arm; never remove failed sessions from the denominator. A failed session may have incurred charges before failure; reconcile the provider bills, not just successful result rows.

This pilot tests basic compatibility, not native web grounding, actual Discord/Slack formatting, attachments, full-sized books, or preservation of the complete bots' persona prompts. The static search fixture provides identical evidence to each model; it does not claim to test real search freshness. Repeat with the real `!think`/judge high-effort profile before considering those paths.

## Stage 2: compare with the actual bots

Before collecting live examples, build a fixed, consented/redacted or synthetic set of 24 conversations: three each for casual, routine code/math, hard reasoning, multilingual, bookclub, research tools, personality/formatting, and emotional support. Keep six as a held-out set. Never automatically export memories or private channel history.

Evaluate three arms with the same eligible prompts, tool results, persona/system instructions, and output limits:

1. Current bot policy after the Opus 5.5 migration (including its cheaper heads).
2. Direct Opus 5.5 as an upper-cost reference.
3. FireRouter as a proposed alternative for eligible turns.

Two repetitions yield 144 sessions. Run sequentially, randomize arm order, and checkpoint after each session. Separate cold-cache first turns from warm-cache continuations; do not reset histories between follow-ups. Report per workload, not just a pooled mean. A router wins only if it also beats the current policy, rather than merely discounting Opus.

Record total billed dollars across **Anthropic plus Fireworks**, actual served model for every call, cache reads/writes by TTL, output/reasoning tokens, tool rounds, error rate, full-answer latency (median/p95), and blinded quality. Reconcile request-level meters against provider billing exports. Unknown rates or unattributed models are **unknown cost**, never zero. Native search fees, retries and failed billed requests belong in the total. Bootstrap confidence intervals by conversation rather than treating dependent turns as independent samples.

Suggested acceptance thresholds, agreed before looking at results: at least 20% lower total cost than the current policy on the candidate workload, no more than 0.2/4 mean quality loss, no increase in critical failures, and p95 completion latency no worse than 1.2x baseline. A 24-conversation set is a screening exercise, not enough evidence to certify rare failure rates. Expand the held-out evaluation if intervals are inconclusive.

## Integration decision

Keep explicit model commands, research-panel member identities, and companion distress/support turns pinned during any first trial. Sending an explicit `!claude` turn to an unidentified GLM response would violate the user's model selection. Likewise, replacing several panelists with the same opaque router could reduce panel diversity.

If the evaluation passes, introduce a separate opt-in experimental provider/command with a distinct canonical identity. Attribute cost and energy to the **served** model per call, using unknown estimates where metadata is absent. Preserve signed thinking blocks and tool history. Review ISAIC's residency/retention requirements for every possible route; Fireworks-hosted open-model assurances do not automatically cover an external Anthropic route.

Only after a successful isolated trial consider ordinary auto-routed turns. Keep an immediate config rollback and do not change the participation gate or model aliases as part of the experiment.

Fireworks' advertised 57% saving and 98.1% relative accuracy came from its own coding traffic. They are useful motivation, not evidence about these bots. [Announcement and methodology](https://fireworks.ai/blog/introducing-firerouter-with-opus)
