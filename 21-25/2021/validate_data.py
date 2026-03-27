#!/usr/bin/env python3
"""
Validate extracted JSON data against known cell values from the Excel files.
Performs spot-checks, row counts, type checks, and range sanity checks.
"""

import json
import os
import sys

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# Expected day counts per month
EXPECTED_DAYS = {
    "july": 31,
    "august": 31,
    "september": 30,
    "october": 31,
    "november": 30,
    "december": 31,
}

# Spot-checks: (month, day_index, field, expected_value)
# These are manually verified values from the Excel files
SPOT_CHECKS = [
    # July
    ("july", 0, "date", "2021-07-01"),
    ("july", 0, "inlet_ph", 6.67),
    ("july", 0, "inlet_bod", 300),
    ("july", 0, "inlet_cod", 640),
    ("july", 0, "inlet_tss", 405),
    ("july", 0, "power_ge", 0.8),
    ("july", 0, "power_nea", 7.61),
    ("july", 0, "flow", 15.395),
    ("july", 0, "grit_tss", 210),
    ("july", 0, "primary_ph", 6.68),
    ("july", 0, "primary_tss", 193),
    ("july", 0, "secondary_tss", 12),
    ("july", 0, "effluent_ph", 6.92),
    ("july", 0, "effluent_bod", 6.6),
    ("july", 0, "effluent_tss", 7),
    # July day 2 - scientific notation test
    ("july", 1, "inlet_total_coliform", 500000.0),
    ("july", 1, "inlet_fecal_coliform", 40000.0),
    # July day 1 - missing values
    ("july", 0, "inlet_tkn", None),
    ("july", 0, "inlet_og", None),
    # July last day
    ("july", 30, "date", "2021-07-31"),
    ("july", 30, "inlet_ph", 6.98),
    ("july", 30, "inlet_bod", 225),
    # August day 1
    ("august", 0, "date", "2021-08-01"),
    ("august", 0, "power_ge", 0.9),
    # September day 1
    ("september", 0, "date", "2021-09-01"),
    # October day 1
    ("october", 0, "date", "2021-10-01"),
    ("october", 0, "power_ge", 1.4),
    # November day 1
    ("november", 0, "date", "2021-11-01"),
    ("november", 0, "power_ge", 4.5),
    # December day 1
    ("december", 0, "date", "2021-12-01"),
    ("december", 0, "power_ge", 1.5),
]

# Numeric fields for range/type checks
NUMERIC_FIELDS = [
    "power_ge", "power_nea", "power_total", "power_per_flow",
    "flow", "inlet_ph", "inlet_bod", "inlet_cod", "inlet_tss",
    "grit_tss",
    "primary_ph", "primary_tss", "primary_bod", "primary_cod",
    "secondary_ph", "secondary_tss", "secondary_bod", "secondary_cod",
    "sec_sed_ph", "sec_sed_tss", "sec_sed_bod", "sec_sed_cod",
    "effluent_ph", "effluent_bod", "effluent_cod", "effluent_tss",
]

PH_FIELDS = [
    "inlet_ph", "primary_ph", "secondary_ph", "sec_sed_ph", "effluent_ph",
]


def load_month(month):
    path = os.path.join(BASE_DIR, f"{month}.json")
    with open(path) as f:
        return json.load(f)


def check_spot_checks(data_cache):
    """Verify hardcoded known values."""
    passes = 0
    fails = 0
    for month, idx, field, expected in SPOT_CHECKS:
        if month not in data_cache:
            print(f"  FAIL: {month} not loaded")
            fails += 1
            continue
        actual = data_cache[month]["days"][idx].get(field)
        if isinstance(expected, float):
            match = actual is not None and abs(actual - expected) < 0.01
        else:
            match = actual == expected
        if match:
            passes += 1
        else:
            print(f"  FAIL: {month}[{idx}].{field} = {actual!r}, expected {expected!r}")
            fails += 1
    return passes, fails


def check_row_counts(data_cache):
    """Verify each month has the expected number of days."""
    passes = 0
    fails = 0
    for month, expected in EXPECTED_DAYS.items():
        actual = len(data_cache[month]["days"])
        if actual == expected:
            passes += 1
        else:
            print(f"  FAIL: {month} has {actual} days, expected {expected}")
            fails += 1
    return passes, fails


def check_types_and_ranges(data_cache):
    """Verify numeric fields are float/int/None and within sane ranges."""
    passes = 0
    fails = 0
    for month, data in data_cache.items():
        for i, day in enumerate(data["days"]):
            for field in NUMERIC_FIELDS:
                val = day.get(field)
                if val is None:
                    passes += 1
                    continue
                if not isinstance(val, (int, float)):
                    print(f"  FAIL: {month}[{i}].{field} = {val!r} (not numeric)")
                    fails += 1
                    continue
                # Range checks
                if field in PH_FIELDS and not (0 <= val <= 14):
                    print(f"  FAIL: {month}[{i}].{field} = {val} (pH out of 0-14)")
                    fails += 1
                elif val < 0 and field not in PH_FIELDS:
                    print(f"  FAIL: {month}[{i}].{field} = {val} (negative)")
                    fails += 1
                else:
                    passes += 1
    return passes, fails


def check_missing_values(data_cache):
    """Report count of missing values per field per month."""
    print("\n  Missing values summary:")
    all_fields = NUMERIC_FIELDS + [
        "inlet_tkn", "inlet_og", "inlet_po4",
        "inlet_total_coliform", "inlet_fecal_coliform",
        "effluent_og", "effluent_nh3",
        "effluent_total_coliform", "effluent_fecal_coliform", "effluent_frc",
    ]
    header = f"  {'Field':<30}" + "".join(f"{m[:3]:>6}" for m in EXPECTED_DAYS.keys())
    print(header)
    print("  " + "-" * (30 + 6 * len(EXPECTED_DAYS)))
    for field in all_fields:
        row = f"  {field:<30}"
        for month in EXPECTED_DAYS.keys():
            days = data_cache[month]["days"]
            nulls = sum(1 for d in days if d.get(field) is None)
            row += f"{nulls:>6}"
        print(row)


def main():
    print("=" * 60)
    print("DATA VALIDATION REPORT")
    print("=" * 60)

    # Load all months
    data_cache = {}
    for month in EXPECTED_DAYS:
        try:
            data_cache[month] = load_month(month)
        except FileNotFoundError:
            print(f"ERROR: {month}.json not found in {BASE_DIR}")
            sys.exit(1)

    total_pass = 0
    total_fail = 0

    # 1. Spot checks
    print("\n1. SPOT CHECKS (known cell values)")
    p, f = check_spot_checks(data_cache)
    total_pass += p
    total_fail += f
    print(f"   {p} passed, {f} failed")

    # 2. Row counts
    print("\n2. ROW COUNTS")
    p, f = check_row_counts(data_cache)
    total_pass += p
    total_fail += f
    print(f"   {p} passed, {f} failed")

    # 3. Type and range checks
    print("\n3. TYPE & RANGE CHECKS")
    p, f = check_types_and_ranges(data_cache)
    total_pass += p
    total_fail += f
    print(f"   {p} passed, {f} failed")

    # 4. Missing values report
    print("\n4. MISSING VALUES REPORT")
    check_missing_values(data_cache)

    # 5. Control limits
    print("\n5. CONTROL LIMITS")
    for month in ["july"]:
        limits = data_cache[month]["control_limits"]
        for k, v in limits.items():
            status = "✓" if v != "N/A" else "⚠ N/A"
            print(f"   {k:<30} {str(v):<20} {status}")

    # Summary
    print("\n" + "=" * 60)
    print(f"TOTAL: {total_pass} passed, {total_fail} failed")
    if total_fail > 0:
        print("STATUS: ❌ FAILED")
        sys.exit(1)
    else:
        print("STATUS: ✅ ALL PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()
