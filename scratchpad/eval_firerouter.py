"""Synthetic, standalone Opus 5.5 vs FireRouter compatibility/quality pilot.

Dry-run by default. See docs/firerouter-evaluation.md. No bot state is imported.
"""
import argparse
import csv
import json
import os
import random
import time
from pathlib import Path


BOOK = "\n".join(
    f"Log {i}: Mira kept the observatory open. Sol repaired the blue telescope. "
    "The council proposed closing it, but the children used it every Friday."
    for i in range(32)
) + "\nFinal log: Mira accepted the council's grant only after they removed its admission fee."

CASES = [
    dict(id="casual", system="You are Balthasar, a warm Discord assistant. Be brief.",
         turns=["Finally got the bot running after three hours lol", "Give me a fitting one-line victory toast."],
         rubric="Warm, concise, follows the conversation, no invented personal facts."),
    dict(id="code", system="Help with Python. Explain the smallest correct fix.",
         turns=["Why does [x for x in [0, None, 2] if x] drop zero? Keep everything except None.",
                "What result should that expression produce?"], rubric="Uses `is not None`; result [0, 2]."),
    dict(id="math", system="Explain arithmetic carefully and briefly.",
         turns=["A request has 1M uncached input tokens, 2M cached tokens, and 100k output tokens. Rates are $4, $0.20, and $20 per million. Total?"],
         rubric="Total $6.40; cache hits are a separate bucket and not double-counted."),
    dict(id="chinese", system="Reply in the language used by the user.",
         turns=["用两句话解释为什么缓存命中可以降低聊天机器人的成本。"], rubric="Two clear Chinese sentences explaining reused input and lower cost."),
    dict(id="french", system="You are a concise French tutor.",
         turns=["Corrige : Si j'aurais su, je serais venu. Explique en français."],
         rubric="Si j'avais su, je serais venu; explains plus-que-parfait after si."),
    dict(id="bookclub", system=[{"type": "text", "text": BOOK, "cache_control": {"type": "ephemeral"}},
                                {"type": "text", "text": "Discuss only this fictional source; label interpretation as interpretation."}],
         turns=["Why did Mira initially resist the grant?", "Did she ultimately reject it? Cite the final log."],
         rubric="Does not invent initial motives; says she accepted after the admission fee was removed."),
    dict(id="research", system="Verify current product claims with web_search. Cite only the returned source.",
         turns=["Did ObservatoryDB 7.2 add offline exports? Verify instead of guessing."], tool=True,
         rubric="Calls web_search; reports yes from the synthetic release note; cites only its URL."),
    dict(id="support", system="You are a calm, supportive companion. Validate without inventing a diagnosis or relationship.",
         turns=["I worked all day on this and it still failed. I feel useless.",
                "Please just sit with me for a minute, no advice."],
         rubric="Validates frustration; respects no-advice request; no diagnosis or grand promises."),
]
TOOL = {"name": "web_search", "description": "Search a synthetic release-note fixture.",
        "input_schema": {"type": "object", "properties": {"query": {"type": "string"}},
                         "required": ["query"], "additionalProperties": False}}
TOOL_RESULT = json.dumps({"title": "ObservatoryDB 7.2 release notes (synthetic)",
                          "url": "https://example.com/observatorydb/7.2",
                          "text": "Version 7.2 adds offline exports."})


def visible(response):
    return "".join(b.text for b in response.content if b.type == "text").strip()


def run_session(client, requested_model, case, max_tokens):
    """Independent append-only conversation per arm, including tool continuations."""
    messages, outputs, calls = [], [], []
    for prompt in case["turns"]:
        messages.append({"role": "user", "content": prompt})
        final_text = ""
        for round_index in range(4):
            started = time.monotonic()
            try:
                response = client.messages.create(
                    model=requested_model, max_tokens=max_tokens, system=case["system"],
                    messages=messages, thinking={"type": "adaptive"},
                    extra_body={"output_config": {"effort": "low"}},
                    **({"tools": [TOOL]} if case.get("tool") else {}),
                )
            except Exception as exc:
                # Preserve earlier billed calls when a later continuation fails.
                outputs.append({"prompt": prompt, "answer": final_text, "complete": False})
                return {"turns": outputs, "calls": calls, "error_type": type(exc).__name__,
                        "status_code": getattr(exc, "status_code", None)}
            tools = [b for b in response.content if b.type == "tool_use"]
            calls.append({"served_model": response.model, "requested_model": requested_model,
                          "latency_seconds": time.monotonic() - started,
                          "stop_reason": response.stop_reason, "usage": response.usage.model_dump(),
                          "tool_calls": [b.model_dump() for b in tools],
                          "request_id": getattr(response, "_request_id", None)})
            # No provider secrets or full signed thinking blocks go into reports.
            messages.append({"role": "assistant", "content": response.content})
            final_text += visible(response)
            if response.stop_reason != "tool_use":
                break
            if not tools:
                outputs.append({"prompt": prompt, "answer": final_text, "complete": False})
                return {"turns": outputs, "calls": calls, "error_type": "MissingToolCalls"}
            if round_index == 3:
                outputs.append({"prompt": prompt, "answer": final_text, "complete": False})
                return {"turns": outputs, "calls": calls, "error_type": "ToolRoundLimit"}
            results = []
            for tool in tools:
                valid = (case.get("tool") and tool.name == "web_search"
                         and isinstance(tool.input, dict) and isinstance(tool.input.get("query"), str))
                results.append({"type": "tool_result", "tool_use_id": tool.id,
                                "content": TOOL_RESULT if valid else "Unknown tool or invalid query",
                                "is_error": not valid})
            messages.append({"role": "user", "content": results})
        outputs.append({"prompt": prompt, "answer": final_text,
                        "complete": response.stop_reason == "end_turn" and bool(final_text)})
    return {"turns": outputs, "calls": calls}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Send the synthetic cases to the APIs (billable).")
    parser.add_argument("--provider-key-configured", action="store_true",
                        help="An Anthropic Provider Key is already connected in Fireworks.")
    parser.add_argument("--forward-anthropic-key", action="store_true",
                        help="Explicitly allow forwarding ANTHROPIC_API_KEY to Fireworks for this pilot.")
    parser.add_argument("--env-file", type=Path, help="Optional .env path; read only with --live.")
    parser.add_argument("--cases", default=",".join(c["id"] for c in CASES))
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--out", type=Path, default=Path("scratchpad/firerouter-results"))
    args = parser.parse_args()
    wanted = args.cases.split(",")
    cases = [c for c in CASES if c["id"] in wanted]
    if set(wanted) - {c["id"] for c in cases} or not cases:
        parser.error("Unknown or empty case selection")
    if not 256 <= args.max_tokens <= 4096:
        parser.error("max-tokens must be between 256 and 4096")
    print(f"{len(cases)} synthetic sessions x 2 arms; no production messages or bot state.")
    print("Maximum API calls including credential probe:", 1 + sum(len(c["turns"]) for c in cases) * 4 * 2)
    if not args.live:
        print("DRY RUN. Cases:", ", ".join(c["id"] for c in cases))
        print("No credentials read and no requests sent. See docs/firerouter-evaluation.md.")
        return
    if not (args.provider_key_configured or args.forward_anthropic_key):
        parser.error("Choose --provider-key-configured or --forward-anthropic-key; otherwise the router could be open-only.")
    if args.env_file:
        from dotenv import load_dotenv
        load_dotenv(args.env_file, override=False)
    import anthropic
    clients = {"opus": anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"], max_retries=0, timeout=120),
               "router": anthropic.Anthropic(api_key=os.environ["FIREWORKS_API_KEY"],
                   base_url="https://api.fireworks.ai/inference", max_retries=0, timeout=120,
                   default_headers={"x-anthropic-api-key": os.environ["ANTHROPIC_API_KEY"]}
                   if args.forward_anthropic_key else {})}
    # Verify the closed route first: Fireworks can silently fall back to open models.
    probe = clients["router"].messages.create(model="firerouter/opus", max_tokens=256,
        messages=[{"role": "user", "content": "Reply with OK."}],
        extra_headers={"x-routing-preference": "1"})
    if probe.model != "claude-opus-5-5":
        raise RuntimeError(f"Expected Opus 5.5 for the forced primary probe; got {probe.model!r}")
    args.out.mkdir(parents=True, exist_ok=True)
    # Unique run directories prevent overwriting a prior evaluation.
    import tempfile
    out = Path(tempfile.mkdtemp(prefix="run-", dir=args.out))
    records, review = [], []
    records.append({"case": "credential_probe", "arm": "router", "served_model": probe.model,
                    "usage": probe.usage.model_dump()})
    rng = random.Random(55)
    for case in cases:
        arms = ["opus", "router"]
        rng.shuffle(arms)
        for slot, arm in enumerate(arms):
            blind_id = f"{case['id']}-{slot + 1}"
            try:
                result = run_session(clients[arm], "claude-opus-5-5" if arm == "opus" else "firerouter/opus", case, args.max_tokens)
            except Exception as exc:
                # Avoid error bodies/headers that could contain keys or request data.
                result = {"error_type": type(exc).__name__, "status_code": getattr(exc, "status_code", None)}
            records.append({"case": case["id"], "arm": arm, "blind_id": blind_id, **result})
            review.append({"id": blind_id, "rubric": case["rubric"], "outputs": json.dumps(result.get("turns", []), ensure_ascii=False),
                           "error_type": result.get("error_type", ""),
                           "quality_0_to_4": "", "critical_failure": "", "notes": ""})
            # Save after every session; preserve completed results if interrupted.
            (out / "results.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out / "blind-review.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(review[0]))
        writer.writeheader()
        writer.writerows(review)
    for client in clients.values():
        client.close()
    print("Saved", out)
    print("Pilot only. Grade blind-review.csv before opening results.json; reconcile usage with both provider bills.")


if __name__ == "__main__":
    main()
