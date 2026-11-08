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
import re
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
      "Order-of-magnitude" in out)
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
      "**Detail**" in out and " req, " in out)
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

print("\nthe volume-free per-Mtok ranking (2026-11-07)")
# The cumulative CO2e column is volume x intensity, and volume is a routing
# artifact. The per-Mtok line is the part that is a property of the model, so
# it must be ordered independently of how many tokens each head was sent.
check("the per-Mtok line is present", "Per Mtok" in out)
mtok = [l for l in out.splitlines() if "Per Mtok" in l][0]
check("it says plainly that it is volume-free", "volume-free" in mtok)
check("Deepseek precedes Claude per token, the reverse of the CO2e column",
      mtok.index("Deepseek") < mtok.index("Claude")
      and right[0].split()[1] == "Deepseek")
# Mistral on the France grid is the cleanest thing in the fleet per token even
# though it is last in BOTH cumulative columns -- the exact inversion the line
# exists to show.
# out2 below adds Mistral; it is last in BOTH cumulative columns and first
# per token (France grid ~20 g/kWh) -- the exact inversion the line exists for.
m3 = [l for l in mgr().get_cost_summary([claude, deep, tiny]).splitlines()
      if "Per Mtok" in l][0]
check("Mistral ranks first per token despite being last cumulatively",
      m3.split("(volume-free): ")[1].startswith("Mistral"))
cl = copy.deepcopy(bot.CLAUDE_PROVIDER)
ds = copy.deepcopy(bot.DEEPSEEK_PROVIDER)
check("intensity is just Wh/1k x grid (Claude 0.6 x 400 = 240 g/Mtok)",
      abs(cl.est_wh_per_1k_tokens * cl.grid_gco2_per_kwh - 240.0) < 0.01)
check("and is independent of traffic (same figure at 10x the tokens)",
      "Claude 240 g" in mtok
      and "Claude 240 g" in mgr().get_cost_summary(
          [provider(bot.CLAUDE_PROVIDER, 1220, 47_288_840, 1_444_060),
           provider(bot.DEEPSEEK_PROVIDER, 4, 2_000, 100)]))

print("\nthe per-dollar ranking (2026-11-07)")
# The dual of per-Mtok: that one is the metric when the WORK is fixed, this one
# when the BUDGET is. They disagree because a dollar buys wildly different
# token counts, so a cheap head on a dirty grid is the worst thing to spend a
# dollar on while being among the cleanest things to send a token to.
pd = [l for l in out.splitlines() if "Per $" in l][0]
check("the per-dollar line is present", bool(pd))
check("it says which direction is good", "lower better" in pd)
check("Claude is better per dollar than Deepseek, inverting per-Mtok",
      pd.index("Claude") < pd.index("Deepseek")
      and mtok.index("Deepseek") < mtok.index("Claude"))
# Needs three providers to show: with two, several orderings coincide.
three = mgr().get_cost_summary([claude, deep, tiny])


def _order(line, lead):
    body = line.split(lead)[1]
    return [w for w in body.replace("·", " ").split()
            if w in ("Claude", "Deepseek", "Mistral")]


t_cost = [r[:26].strip().split()[1]
          for r in three.split("```")[1].strip().splitlines()[1:] if r.strip()]
t_mtok = _order([l for l in three.splitlines() if "Per Mtok" in l][0],
                "(volume-free): ")
t_pd = _order([l for l in three.splitlines() if "Per $" in l][0],
              "(lower better): ")
check("per-$ is a THIRD ordering, matching neither cost nor per-Mtok",
      len(t_pd) == 3 and t_pd != t_cost and t_pd != t_mtok)
check("concretely: cheapest-per-dollar is Mistral, dirtiest is Deepseek",
      t_pd[0] == "Mistral" and t_pd[-1] == "Deepseek")
check("while the cost column still leads with Claude", t_cost[0] == "Claude")

# Realised, not rate-card: cache hits make a dollar buy MORE tokens, so a
# heavier hit rate is worse per dollar on the same grid and the same rates.
cold = provider(bot.DEEPSEEK_PROVIDER, 10, 1_000_000, 10_000)
warm = provider(bot.DEEPSEEK_PROVIDER, 10, 100_000, 10_000, cached=900_000)
cold_pd = [l for l in mgr().get_cost_summary([cold]).splitlines()
           if "Per $" in l][0]
warm_pd = [l for l in mgr().get_cost_summary([warm]).splitlines()
           if "Per $" in l][0]


def _num(line):
    v = line.split("Deepseek")[1].split("g")[0].strip()
    return float(v[:-1]) * 1000 if v.endswith("k") else float(v.replace(",", ""))


check("a cache-heavy provider is WORSE per dollar than a cold one",
      _num(warm_pd) > _num(cold_pd))

# Cache STORAGE bought no tokens, so it must not flatter the denominator.
nostore = provider(bot.GEMINI_PROVIDER, 23, 607_675, 12_909, 1_458_757)
withstore = provider(bot.GEMINI_PROVIDER, 23, 607_675, 12_909, 1_458_757,
                     storage=11.78)
g1 = [l for l in mgr().get_cost_summary([nostore]).splitlines()
      if "Per $" in l][0]
g2 = [l for l in mgr().get_cost_summary([withstore]).splitlines()
      if "Per $" in l][0]
check("cache storage is excluded from the per-dollar denominator", g1 == g2)

# Undefined, not zero, where there is no per-token price.
loc2 = provider(bot.DEEPSEEK_PROVIDER, 5, 100_000, 5_000)
loc2.cost_mode = "local"
lout = mgr().get_cost_summary([loc2])
check("a local backend is omitted rather than shown as perfectly clean",
      "Per $" not in lout)
pl2 = provider(bot.QWEN_PROVIDER, 5, 100_000, 5_000)
pl2.cost_mode, pl2.plan_monthly_usd = "plan", 3.0
check("so is a prepaid plan (its tokens cost no marginal dollar)",
      "Per $" not in mgr().get_cost_summary([pl2]))
check("but a priced provider alongside them still ranks",
      "Per $" in mgr().get_cost_summary([loc2, claude]))

# Formatting: the spread runs from single digits to thousands.
big = provider(bot.DEEPSEEK_PROVIDER, 500, 6_000_000, 170_000, 48_000_000)
check("a four-figure ratio renders compactly as 'k'",
      "k g" in [l for l in mgr().get_cost_summary([big]).splitlines()
                if "Per $" in l][0])
clean = provider(bot.MISTRAL_PROVIDER, 500, 5_000_000, 200_000)
cl_pd = [l for l in mgr().get_cost_summary([clean]).splitlines()
         if "Per $" in l][0]
check("and a sub-gram ratio reads '<1', never '0'",
      ("<1" in cl_pd) or (" 0 g" not in cl_pd))

print("\ngrid constants are location-based, not REC-adjusted (2026-11-07)")
check("Claude sits at the US average, not the old REC-justified 250",
      cl.grid_gco2_per_kwh == 400.0)
check("China is above the US but not by the old ~2x (national avg ~530)",
      1.2 < ds.grid_gco2_per_kwh / cl.grid_gco2_per_kwh < 1.5)
check("per token Claude is DIRTIER than Deepseek, the fleet's cheap head",
      cl.est_wh_per_1k_tokens * cl.grid_gco2_per_kwh
      > ds.est_wh_per_1k_tokens * ds.grid_gco2_per_kwh)

print("\nDetail never renders a mass as a bare '0 g'")
fr = provider(bot.MISTRAL_PROVIDER, 1, 500, 20)
dout = mgr().get_cost_summary([fr]).split("**Detail**")[1]
check("a sub-gram provider reads '<1 g', not '0 g'",
      "<1 g" in dout and "0 g CO2e" not in dout)

print("\nthe 2,000-char guard")
# Everything that can lengthen a line at once: 7 heads, peak surcharge, tool
# calls and a prepaid plan. Detail is what gives way, never the rankings.
heavy = [provider(c, 50, 500_000, 20_000, 100_000) for c in (
    bot.CLAUDE_PROVIDER, bot.DEEPSEEK_PROVIDER, bot.GEMINI_PROVIDER,
    bot.MISTRAL_PROVIDER, bot.QWEN_PROVIDER, bot.GLM_PROVIDER,
    bot.KIMI_PROVIDER)]
for h in heavy:
    h.total_peak_surcharge = 0.1234
    h.record_tool_requests(search=250, fetch=125)
heavy[4].cost_mode, heavy[4].plan_monthly_usd = "plan", 3.0
hout = mgr().get_cost_summary(heavy)
check(f"the worst case is held under the limit ({len(hout)} chars)",
      len(hout) < 2000)
check("and it is Detail that gave way, not the rankings",
      "omitted" in hout and "by cost" in hout and "Per Mtok" in hout)
check("some Detail still survives rather than all of it vanishing",
      hout.count("\n  **") > 0)
check("an ordinary summary is NOT truncated",
      "omitted" not in out and "**Detail**" in out)

print("\nthe live 8-provider fleet (2026-11-08)")
# The real fleet is 7 heads PLUS the Haiku participation gate, whose name is
# 24 characters. On its first live run that overflowed the 9-char name field,
# slid the right-hand column, and welded a row number onto a share ("<1%8
# Mistral"); the whole Detail section also vanished rather than showing the
# providers worth reading.
gate = provider(bot.CLAUDE_PROVIDER, 40, 60_000, 2_000)
gate.name = "Haiku participation gate"
gate.est_wh_per_1k_tokens, gate.grid_gco2_per_kwh = 0.2, 400.0
gate.input_cost_per_million, gate.output_cost_per_million = 1.0, 5.0
gate.cached_input_cost_per_million = 0.1
fleet = [provider(c, 50, 500_000, 20_000, 100_000) for c in (
    bot.CLAUDE_PROVIDER, bot.DEEPSEEK_PROVIDER, bot.GEMINI_PROVIDER,
    bot.MISTRAL_PROVIDER, bot.QWEN_PROVIDER, bot.GLM_PROVIDER,
    bot.KIMI_PROVIDER)] + [gate]
eight = mgr().get_cost_summary(fleet)
check(f"8 providers fit Discord's limit ({len(eight)} chars)", len(eight) <= 2000)

table = [r for r in eight.split("```")[1].strip().splitlines()[1:] if r.strip()]
check("every ranking row is the same width (the alignment bug)",
      len({len(r) for r in table}) == 1)
check("all 8 rows are present", len(table) == 8)
check("the long name is truncated, not left to overflow",
      any("\u2026" in r for r in table)
      and not any("Haiku participation gate" in r for r in table))
check("no row welds a number onto the previous share",
      not any(re.search(r"%\d", r) for r in table))
check("the canonical provider.name is NOT mutated by display",
      gate.name == "Haiku participation gate")
check("and the full name still appears in the prose rankings",
      "Haiku participation gate" in
      [l for l in eight.splitlines() if "Per Mtok" in l][0])

print("\nDetail degrades gracefully instead of vanishing")
kept = [l for l in eight.splitlines() if l.startswith("  **")]
check("some per-provider detail survives", len(kept) > 0)
check("it is the highest-spend providers that are kept",
      kept[0].startswith("  **Claude**"))
note = [l for l in eight.splitlines() if "more provider line(s) omitted" in l]
check("and the omission is declared", len(note) == 1)
# No cache storage in this fixture, so one detail line == one provider.
claimed = int(re.search(r"\((\d+) more", note[0]).group(1))
check("the declared count matches what was actually dropped",
      claimed == len(fleet) - eight.count("\n  **"))
check("and nothing was dropped silently",
      claimed > 0 and eight.count("\n  **") + claimed == len(fleet))

print("\neven a pathological fleet cannot exceed the limit")
many = []
for i in range(20):
    q = provider(bot.CLAUDE_PROVIDER, 10, 100_000, 5_000)
    q.name = f"Very Long Provider Name {i}"
    many.append(q)
huge = mgr().get_cost_summary(many)
check(f"20 providers still fit ({len(huge)} chars)", len(huge) <= 2000)
check("and the rankings survive even then", "by cost" in huge)

print("\nDiscord markdown cannot eat an identifier")
# A bare underscore is an italics delimiter, and AMORTIZE_TRAINING's paired
# with a marker on a later line, swallowing both and printing
# "AMORTIZETRAINING".
amort = [l for l in out.splitlines() if "amortized training" in l]
if amort:
    check("AMORTIZE_TRAINING is inside a code span",
          "`AMORTIZE_TRAINING=0`" in amort[0])

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
