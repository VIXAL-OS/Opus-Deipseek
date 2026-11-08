"""Offline checks for cost_mode="plan" — prepaid subscriptions (2026-11-07).

Alibaba Model Studio's Token Plan ($3–5/mo of Credits) bills a flat monthly fee
and draws usage from a prepaid allowance, so reporting a per-token $ would
double-count against the fee. The load-bearing requirement is that this costs us
NOTHING in accounting fidelity elsewhere: token counts, 🌱 energy and CO₂ are
computed from the token counters alone and must come out BYTE-IDENTICAL to
metered mode. That is what makes a plan safe to model at all.

Deliberately NOT modelled: a credits counter. The OpenAI-compatible endpoint
returns tokens, not credits, and the token→credit ratio is Alibaba's to define,
so a local conversion would be invented precision.

Each registry check runs in its own subprocess: ProviderRegistry.from_config
mutates the module-level provider constants in place, so config scenarios
pollute each other within one interpreter.

No network, no keys. Run from the repo root:

    PYTHONIOENCODING=utf-8 python scratchpad/test_plan_cost_mode.py

Imports `bot` by default; set BOT_MODULE=isaic.core to run against the Slack
bot's copy (from the isaic-slack-bot repo root).
"""
import copy
import importlib
import os
import subprocess
import sys
import textwrap

sys.path.insert(0, os.getcwd())
MODULE = os.getenv("BOT_MODULE", "bot")
bot = importlib.import_module(MODULE)

ok = fail = 0


def check(name, cond):
    global ok, fail
    print(("  ok   " if cond else "  FAIL ") + name)
    ok += bool(cond)
    fail += (not cond)


def in_subprocess(body: str) -> str:
    """Run a snippet in a fresh interpreter; return its combined output."""
    src = f"import {MODULE} as bot\n" + textwrap.dedent(body)
    # encoding/errors explicitly: the registry prints emoji status lines, and a
    # Windows locale decode would otherwise blow up or hand back None.
    r = subprocess.run(
        [sys.executable, "-c", src], capture_output=True,
        encoding="utf-8", errors="replace",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    return (r.stdout or "") + (r.stderr or "")


def sample(mode, fee=0.0):
    p = copy.deepcopy(bot.QWEN_PROVIDER)
    p.cost_mode = mode
    p.plan_monthly_usd = fee
    p.peak_windows_utc = ()
    p.record_usage(input_tokens=1_000_000, output_tokens=500_000, cached_input_tokens=250_000)
    return p


print("the default is unchanged")
check("providers ship as metered", bot.QWEN_PROVIDER.cost_mode == "metered")
check("and carry no plan fee", bot.QWEN_PROVIDER.plan_monthly_usd == 0.0)

print("\nplan mode drops per-token $ but keeps the tokens")
metered, plan = sample("metered"), sample("plan", 3.0)
check("metered reports a real per-token cost", metered.get_cost() > 0)
check("plan reports no per-token cost", plan.get_cost() == 0.0)
check("plan still counted every input token",
      plan.total_input_tokens == metered.total_input_tokens == 1_000_000)
check("plan still counted output + cached tokens",
      plan.total_output_tokens == 500_000 and plan.total_cached_input_tokens == 250_000)
check("plan still counts requests", plan.total_requests == metered.total_requests)

print("\nenergy and CO2 are BYTE-IDENTICAL to metered — the whole point")
check("energy Wh identical",
      plan.get_energy_wh() == metered.get_energy_wh() > 0)
check("CO2 g identical",
      plan.get_co2_g(400.0) == metered.get_co2_g(400.0) > 0)
check("amortized training CO2 identical",
      plan.get_amortized_train_co2_g(1e13) == metered.get_amortized_train_co2_g(1e13))

print("\nthe flat fee is NOT usage")
check("the fee does not leak into get_cost()", sample("plan", 5.0).get_cost() == 0.0)
check("the fee is carried for the footer", sample("plan", 5.0).plan_monthly_usd == 5.0)
summary_src = __import__("inspect").getsource(bot.ConversationManager.get_cost_summary)
check("!cost labels a plan line", "prepaid plan" in summary_src)
check("!cost footers the fee separately", "prepaid plans" in summary_src)
check("the footer says it is excluded from the total",
      "not in the usage" in summary_src)

print("\nper-request tool spend still bills under a plan")
p = sample("plan", 3.0)
p.search_cost_per_1k_requests = 0.75
p.record_tool_requests(search=1000)
check("a hosted per-request tool is billed even on a plan",
      abs(p.get_cost() - 0.75) < 1e-12)

print("\nlocal mode is untouched by the change")
loc = sample("local")
check("local still reports no per-token $", loc.get_cost() == 0.0)
check("local still reports energy", loc.get_energy_wh() > 0)
# The per-provider rows moved to !costdetail on 2026-11-08, so the
# electricity-only label lives in that method now. Assert the source was
# actually found, so a future rename fails loudly instead of passing on
# an empty search.
detail_src = __import__("inspect").getsource(
    bot.ConversationManager.get_cost_detail)
check("the detail method was found at all", len(detail_src) > 500)
check("!costdetail still labels local electricity-only",
      "electricity only" in detail_src)

print("\nconfig wiring (fresh interpreter each)")
out = in_subprocess("""
    r = bot.ProviderRegistry.from_config(
        {"providers": {"qwen": {"cost_mode": "plan", "plan_monthly_usd": 3}}})
    q = r.by_id("qwen")
    print("RESULT", q.cost_mode, q.plan_monthly_usd)
""")
check("config can set cost_mode=plan + fee", "RESULT plan 3.0" in out)

out = in_subprocess("""
    r = bot.ProviderRegistry.from_config({"providers": {"qwen": {"cost_mode": "bogus"}}})
    print("RESULT", r.by_id("qwen").cost_mode)
""")
check("an unknown cost_mode warns and keeps metered",
      "ignoring unknown cost_mode" in out and "RESULT metered" in out)

out = in_subprocess("""
    r = bot.ProviderRegistry.from_config({"providers": {"qwen": {"plan_monthly_usd": "x"}}})
    print("RESULT", r.by_id("qwen").plan_monthly_usd)
""")
check("a non-numeric fee warns and is ignored",
      "is not a number" in out and "RESULT 0.0" in out)

out = in_subprocess("""
    bot.ProviderRegistry.from_config({"providers": {"qwen": {"plan_monthly_usd": 3}}})
""")
check("a fee without cost_mode='plan' warns that it won't show",
      "cost_mode is 'metered'" in out)

out = in_subprocess("""
    r = bot.ProviderRegistry.from_config({})
    print("RESULT", r.by_id("qwen").cost_mode, r.by_id("qwen").plan_monthly_usd)
""")
check("no config block → byte-for-byte the old behavior",
      "RESULT metered 0.0" in out)

print("\nbackends can carry it too")
check("plan_monthly_usd is backend-overridable",
      "plan_monthly_usd" in bot._BACKEND_FIELD_MAP)
check("cost_mode is backend-overridable", "cost_mode" in bot._BACKEND_FIELD_MAP)

print(f"\n{ok} passed, {fail} failed")
sys.exit(1 if fail else 0)
