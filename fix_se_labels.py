"""
fix_se_labels.py - Replace old sub-experiment labels with SE convention in display text only.
Protects Python code patterns (exp_key comparisons, dict keys used as data identifiers).
"""
import re

TARGET = "modeling/scripts/generate_unified_report.py"

with open(TARGET, "r") as f:
    lines = f.readlines()

# Ordered from most specific to least specific to avoid partial matches
DISPLAY_MAP = [
    ("Exp3-S3-FS",    "Exp3-SE4-FS"),
    ("Exp3-S3",       "Exp3-SE3"),
    ("Exp3-S2-FS",    "Exp3-SE2-FS"),
    ("Exp3-S2",       "Exp3-SE2"),
    ("Exp3-S1",       "Exp3-SE1"),
    ("Exp4-S1",       "Exp4-SE1"),
    ("Exp4-S2",       "Exp4-SE2"),
    ("Exp5-S1",       "Exp5-SE1"),
    ("Exp5-S2",       "Exp5-SE2"),
    ("Exp2-Sub2-FS",  "Exp2-SE3-FS"),
    ("Exp2-Sub2-Cyc", "Exp2-SE2-Cyc"),
    ("Exp2-Sub2",     "Exp2-SE2"),
    ("Exp2-Sub1-Clr", "Exp2-SE1-Clr"),
    ("Exp2-Sub1-Sed", "Exp2-SE1-Sed"),
    ("Exp2-Sub1-FS",  "Exp2-SE1-FS"),
    ("Exp2-Sub1",     "Exp2-SE1"),
    ("Exp1-Sub1",     "Exp1-SE1"),
]

# Lines that contain these patterns are Python code and must NOT be changed
PROTECTED_LINE_PATTERNS = [
    # exp_key comparisons
    re.compile(r'==\s*"Exp'),
    re.compile(r'!=\s*"Exp'),
    # dict key definitions (old_key: or "old_key":)
    re.compile(r'^\s*"Exp[^"]*"\s*:'),
    re.compile(r'^\s*\("Exp[^"]*"'),       # tuple entries in lists like _DS_EXP_MAP
    # function return / variable assignment with exp_key string
    re.compile(r'exp_key\s*=\s*"Exp'),
    # _section_bests_json entries (dict keys)
    re.compile(r'"exp\d+-s'),
    # isin / list membership checks
    re.compile(r'\["Exp'),
    re.compile(r'isin\(\['),
    # ann_extra norm function (exp key mapping)
    re.compile(r'_norm_ann_extra'),
    # EXP_CHART_LABELS, EXP_SOURCE_LABELS dict entries (keys handled separately)
    re.compile(r'"(Exp[^"]+)"\s*:\s*"'),   # key: value string pairs
    # load_all_data tuples
    re.compile(r'\("exp\d+'),
]

def is_protected(line):
    for pat in PROTECTED_LINE_PATTERNS:
        if pat.search(line):
            return True
    return False

changed = 0
out_lines = []
for i, line in enumerate(lines):
    if is_protected(line):
        out_lines.append(line)
        continue
    new_line = line
    for old, new in DISPLAY_MAP:
        if old in new_line:
            new_line = new_line.replace(old, new)
    if new_line != line:
        changed += 1
        print(f"  Line {i+1}: {line.rstrip()!r}")
        print(f"       → {new_line.rstrip()!r}")
    out_lines.append(new_line)

with open(TARGET, "w") as f:
    f.writelines(out_lines)

print(f"\nDone. {changed} lines changed.")
