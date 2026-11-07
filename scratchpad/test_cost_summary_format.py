"""Offline checks for the ranked !cost summary (2026-11-07).

The point of the rewrite is that cost and carbon rank DIFFERENTLY, and a
registry-ordered list hid it. These checks pin that the two rankings are
independent, that the carbon figure carries everyday equivalents, and that the
whole thing still fits a Discord message.

No network, no keys. Run from the repo root:

    PYTHONIOENCODING=utf-8 python scratchpad/test_cost_summary_format.py

Imports `bot` by default; set BOT_MODULE=isaic.core for the Slack bot's copy.
"""
import copy
import importlib
import os
import sys

sys.path.insert(0, os.getcwd())
bot = importlib.import_module(os.getenv("BOT_MODULE", "bot"))

ok = fail = 0


def check(name, cond):
    global ok, fail
    print(("  ok   " if cond else "  FAIL ") + name)
    ok += bool(cond)
    fail += (not cond)


def mgr():
    m = bot.ConversationManager.__new__(bot.ConversationManager)
    m.reading_materials = {}
    return m


def provider(const, requests, uncached, out, cached=0, storage=0.0):
    p = copy.deepcopy(const)
    p.total_requests = requests
    p.total_input_tokens = uncached
    p.total_output_tokens = out
    p.total_cached_input_tokens = cached
    p.total_cache_storage_cost_est = storage
    p.total_token_cost = None
    p.peak_windows_utc = ()
    return p


print("empty state")
check("no traffic -> the short message, not an empty table",
      mgr().get_cost_summary([]) == "💰 No API calls made yet.")
check("a provider with zero requests is skipped entirely",
      mgr().get_cost_summary([provider(bot.CLAUDE_PROVIDER, 0, 0, 0)])
      == "💰 No API calls made yet.")

print("\nthe rankings are independent — the whole point of the rewrite")
# Claude: dear, clean grid. DeepSeek: cheap, dirty grid and far more tokens.
# This is the real shape of the fleet, and the orders must come out different.
claude = provider(bot.CLAUDE_PROVIDER, 122, 4_728_884, 144_406, 2_288_714)
deep = provider(bot.DEEPSEEK_PROVIDER, 459, 5_907_676, 166_650, 48_230_912)
out = mgr().get_cost_summary([claude, deep])
table = out.split("```")[1]
rows = [r for r in table.strip().splitlines()[1:] if r.strip()]
left = [r[:26].strip() for r in rows]
right = [r[26:].strip() for r in rows]
check("cost ranking puts Claude first", left[0].split()[1] == "Claude")
check("CO2e ranking puts Deepseek first", right[0].split()[1] == "Deepseek")
check("so the two orders really do differ",
      [l.split()[1] for l in left] != [r.split()[1] for r in right])
check("both columns are present with a header",
      "by cost" in table and "by CO2e" in table)

print("\ncarbon equivalents")
check("the carbon line carries a driving equivalent",
      "petrol car" in out)
check("and a gaming equivalent", "PC gaming" in out)
check("and says plainly that it is order-of-magnitude",
      "Order-of-magnitude only" in out)
check("equivalents helper returns '' for a negligible mass",
      bot._co2_equivalents(0.4) == "")
check("1 kg of CO2e is ~4 km of driving at 250 g/km",
      "4 km" in bot._co2_equivalents(1000.0))
check("a small mass falls back to minutes of gaming, not '0 h'",
      "min" in bot._co2_equivalents(50.0))

print("\nsmall shares read as '<1%', not '0%'")
tiny = provider(bot.MISTRAL_PROVIDER, 3, 18_214, 1_206)
out2 = mgr().get_cost_summary([claude, deep, tiny])
check("a sub-1% row shows <1%", "<1%" in out2)
check("and never a bare 0% that reads as nothing",
      " 0%" not in out2.split("```")[1])

print("\nnothing from the old format was dropped")
check("per-provider detail still lists requests and tokens",
      "**Detail**" in out and "requests," in out)
check("cache hit rate survives", "% hit)" in out)
check("the grand total survives", "**Total**" in out)
gem = provider(bot.GEMINI_PROVIDER, 23, 607_675, 12_909, 1_458_757, storage=11.78)
out3 = mgr().get_cost_summary([gem])
check("Gemini cache storage still itemised",
      "cache storage" in out3 and "11.78" in out3)
loc = provider(bot.DEEPSEEK_PROVIDER, 5, 1000, 100)
loc.cost_mode = "local"
check("local backends still say electricity only",
      "electricity only" in mgr().get_cost_summary([loc]))
plan = provider(bot.QWEN_PROVIDER, 5, 1000, 100)
plan.cost_mode, plan.plan_monthly_usd = "plan", 3.0
pout = mgr().get_cost_summary([plan])
check("prepaid plans still footer below the total",
      "prepaid plan" in pout and "not in the usage total" in pout)
tool = provider(bot.DEEPSEEK_PROVIDER, 5, 1000, 100)
tool.record_tool_requests(search=1000)
check("per-request tool spend still itemised",
      "tool calls" in mgr().get_cost_summary([tool]))

print("\nit still fits a Discord message")
every = [provider(c, 50, 500_000, 20_000, 100_000) for c in (
    bot.CLAUDE_PROVIDER, bot.DEEPSEEK_PROVIDER, bot.GEMINI_PROVIDER,
    bot.MISTRAL_PROVIDER, bot.QWEN_PROVIDER, bot.GLM_PROVIDER,
    bot.KIMI_PROVIDER)]
full = mgr().get_cost_summary(every)
check(f"all 7 heads render in {len(full)} chars (< 2000)", len(full) < 2000)
check("the table block is closed", full.count("```") == 2)

print(f"\n{ok} passed, {fail} failed")
sys.exit(1 if fail else 0)
