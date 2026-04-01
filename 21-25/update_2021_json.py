"""
One-time script: normalise 2021 JSON files to match the 2022-2025 format.

Changes applied to each month file and all_months.json:
  1. power_ge / power_nea / power_total  → multiply by 1000  (MW → KWh)
  2. power_unit added = "KWh"
  3. has_composite added = false
  4. Composite fields added as null for every day:
       inlet_ph_comp, inlet_bod_comp, inlet_cod_comp, inlet_tss_comp
       effluent_ph_comp, effluent_bod_comp, effluent_cod_comp, effluent_tss_comp
"""

import json
import os

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "2021", "data")
POWER_KEYS = ("power_ge", "power_nea", "power_total")
COMP_FIELDS = [
    "inlet_ph_comp", "inlet_bod_comp", "inlet_cod_comp", "inlet_tss_comp",
    "effluent_ph_comp", "effluent_bod_comp", "effluent_cod_comp", "effluent_tss_comp",
]


def patch(data):
    data["power_unit"] = "KWh"
    data["has_composite"] = False
    for day in data.get("days", []):
        for pk in POWER_KEYS:
            if day.get(pk) is not None:
                day[pk] = round(day[pk] * 1000, 3)
        for cf in COMP_FIELDS:
            day.setdefault(cf, None)
    return data


files = [f for f in os.listdir(BASE_DIR) if f.endswith(".json")]
for fname in files:
    path = os.path.join(BASE_DIR, fname)
    with open(path) as f:
        raw = json.load(f)

    # all_months.json is a dict of month_key → month_data
    if fname == "all_months.json":
        for mk in raw:
            patch(raw[mk])
    else:
        patch(raw)

    with open(path, "w") as f:
        json.dump(raw, f, indent=2)
    print(f"  Updated {fname}")

print("Done.")
