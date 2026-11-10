"""Does the 35% -> 53% auto swing follow from the turns that actually happened?

Turns between the 18:31 and 20:12 !cost runs: three plain messages (auto-routed
to DeepSeek) and two !claude overrides. Per-request token sizes come from the
18:31 !costdetail run.
"""
ds_per_req = (5_912_555 + 166_806 + 48_242_432) / 460
cl_per_req = (4_758_407 + 145_531 + 2_288_714) / 123
print(f"DeepSeek ~{ds_per_req:,.0f} tokens/request, "
      f"Claude ~{cl_per_req:,.0f} tokens/request")

auto_added = 3 * ds_per_req
over_added = 2 * cl_per_req
# Solve for the prior attributed total A given the shares moved 0.35 -> 0.53:
#   (0.35A + auto_added) / (A + auto_added + over_added) = 0.53
A = (auto_added - 0.53 * (auto_added + over_added)) / (0.53 - 0.35)
auto, yours = 0.35 * A + auto_added, 0.65 * A + over_added
print(f"implied prior attributed total: {A:,.0f} tokens "
      f"(vs 63.8M unattributed history)")
print(f"predicted after: auto {auto / (auto + yours) * 100:.0f}% / "
      f"yours {yours / (auto + yours) * 100:.0f}%  -- live line says 53% / 47%")

# Why the dollars barely moved: DeepSeek at its 89% cache-hit rate.
unc = ds_per_req * 0.11
cached = ds_per_req * 0.89
out = ds_per_req * 0.0031
ds_cost = unc / 1e6 * 0.15 + cached / 1e6 * 0.003 + out / 1e6 * 0.60
print(f"three DeepSeek turns cost ~${ds_cost * 3:.4f} "
      f"-- invisible against a 2dp $1.13")
print(f"energy added ~{auto_added / 1000 * 0.2:,.0f} Wh against 16,300 "
      f"-- still reads '16.3 kWh'")
print(f"observed total delta ${33.8642 - 33.6829:.4f}, "
      f"Claude delta ${28.72 - 28.54:.2f} -- so ~all of it is Claude's")
