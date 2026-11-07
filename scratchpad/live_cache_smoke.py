"""LIVE smoke test for the 2026-11-07 prompt-cache work. SPENDS REAL MONEY.

Makes three real Opus 5.5 calls (~$0.05 total) to prove the thing the offline
harness cannot: that Anthropic actually caches our prefix and reads it back
when the volatile context churns.

    1. cold   — first call with the breakpoint. Expect a cache WRITE.
    2. warm   — same stable prefix, DIFFERENT volatile tail. Expect a cache
                READ of the same size. This is the whole claim.
    3. control— the OLD shape: volatile context joined into the system prompt.
                Expect no read and a fresh write, i.e. what every turn used to
                do.

Run from the repo root with a funded ANTHROPIC_API_KEY in .env:

    PYTHONIOENCODING=utf-8 python scratchpad/live_cache_smoke.py

Nothing is written to disk and no Discord connection is made.
"""
import asyncio
import os
import sys
import tempfile

sys.path.insert(0, os.getcwd())
import bot  # noqa: E402  (loads .env at import, before the chdir below)

GUILD = 0
ok = fail = 0


def check(name, cond):
    global ok, fail
    print(("  ok   " if cond else "  FAIL ") + name)
    ok += bool(cond)
    fail += (not cond)


def build_bot():
    here = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            return bot.ClaudeBot()
        finally:
            os.chdir(here)


b = build_bot()
p = b.claude_provider
if not b.clients.get("claude") and not getattr(b, "claude_client", None):
    print("ANTHROPIC_API_KEY missing or Claude disabled — nothing to test.")
    sys.exit(1)

# The REAL stable prefix, exactly as a chat turn builds it.
stable = b._build_system_prompt(p, "", guild_id=GUILD)
tools = [{"type": "web_search_20260209", "name": "web_search"}]

counted = b.claude_client.messages.count_tokens(
    model=p.model_id, system=stable, tools=tools,
    messages=[{"role": "user", "content": "hi"}],
)
print(f"prefix: {counted.input_tokens:,} tokens (tools + system + a stub turn)")
print(f"        Anthropic's minimum cacheable prefix is 1,024 on Opus\n")
check("the prefix clears the minimum cacheable length",
      counted.input_tokens > 1024)


def turn(volatile: str, *, old_shape: bool):
    """One real call. Returns (cached_read_delta, uncached_delta)."""
    messages = [{"role": "user", "content": "Sarah: reply with exactly: OK"}]
    system = stable
    if old_shape:
        # Pre-fix placement: churn joined into the system message.
        system = f"{stable}\n\n{volatile}"
    else:
        bot.ClaudeBot._append_volatile_context(messages, volatile)
    before_cached = p.total_cached_input_tokens
    before_in = p.total_input_tokens
    asyncio.run(b._generate_claude_response(
        GUILD, messages, system, cache_system=True, tools=tools,
    ))
    return (p.total_cached_input_tokens - before_cached,
            p.total_input_tokens - before_in)


V1 = "**Other recent threads**\n- **planning** (5m ago)\n\nMEM: sarah likes tea"
V2 = "**Other recent threads**\n- **planning** (6m ago)\n\nMEM: sarah likes tea, and coffee"

print("1. cold call (expect a cache WRITE, no read)")
read1, in1 = turn(V1, old_shape=False)
print(f"   cached read: {read1:,}   uncached in: {in1:,}")

print("2. warm call — thread age AND memory both changed (expect a READ)")
read2, in2 = turn(V2, old_shape=False)
print(f"   cached read: {read2:,}   uncached in: {in2:,}")
check("the warm call READ the cached prefix", read2 > 0)
check("it read roughly the whole prefix, not a fragment",
      read2 >= counted.input_tokens * 0.8)
check("the churning tail did NOT invalidate it", read2 >= read1)

print("\n3. control — volatile back in the system prompt, the OLD shape")
read3, in3 = turn(V2 + " (control)", old_shape=True)
print(f"   cached read: {read3:,}   uncached in: {in3:,}")
check("the old placement read nothing like the full prefix (it re-bills)",
      read3 < counted.input_tokens * 0.8)

print(f"\nthis run's metered Claude spend: ${p.get_cost():.6f}")
print(f"{ok} passed, {fail} failed")
sys.exit(1 if fail else 0)
