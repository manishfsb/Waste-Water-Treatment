"""
fix_se_revert_code.py - Revert the incorrectly changed exp_key IDENTIFIERS back to old names.
Display text (HTML strings, narrative) can stay with SE names.
Only code-level identifiers (dict keys, filter values, _DS_EXP_MAP values) must revert.
"""

TARGET = "modeling/scripts/generate_unified_report.py"

with open(TARGET, "r") as f:
    content = f.read()

# Each tuple: (old_broken_string, correct_string_to_restore)
# These are code-level identifiers that were incorrectly changed.
REVERTS = [
    # _DS_EXP_MAP values (these assign exp_keys when loading data)
    ('("experiment5/sub_exp2",                            "Exp5-SE2")',
     '("experiment5/sub_exp2",                            "Exp5-S2")'),
    ('("experiment5/sub_exp1",                            "Exp5-SE1")',
     '("experiment5/sub_exp1",                            "Exp5-S1")'),
    ('("experiment4/sub_exp2",                            "Exp4-SE2")',
     '("experiment4/sub_exp2",                            "Exp4-S2")'),
    ('("experiment4/sub_exp1",                            "Exp4-SE1")',
     '("experiment4/sub_exp1",                            "Exp4-S1")'),
    ('("experiment3/sub_exp3/feature_selected_datasets", "Exp3-SE4-FS")',
     '("experiment3/sub_exp3/feature_selected_datasets", "Exp3-S3-FS")'),
    ('("experiment3/sub_exp3",                           "Exp3-SE3")',
     '("experiment3/sub_exp3",                           "Exp3-S3")'),
    ('("experiment3/sub_exp2",                           "Exp3-SE2-FS")',
     '("experiment3/sub_exp2",                           "Exp3-S2-FS")'),
    ('("experiment3/sub_exp1",                           "Exp3-SE1")',
     '("experiment3/sub_exp1",                           "Exp3-S1")'),
    ('("experiment2/sub_exp2/feature_selected_datasets", "Exp2-SE3-FS")',
     '("experiment2/sub_exp2/feature_selected_datasets", "Exp2-Sub2-FS")'),
    ('("experiment2/sub_exp2",                           "Exp2-SE2")',
     '("experiment2/sub_exp2",                           "Exp2-Sub2")'),
    ('("experiment2/sub_exp1/feature_selected_datasets", "Exp2-SE1-FS")',
     '("experiment2/sub_exp1/feature_selected_datasets", "Exp2-Sub1-FS")'),
    ('("experiment2/sub_exp1",                           "Exp2-SE1")',
     '("experiment2/sub_exp1",                           "Exp2-Sub1")'),
    ('("experiment1/sub_exp1",                           "Exp1-SE1")',
     '("experiment1/sub_exp1",                           "Exp1-Sub1")'),

    # EXP_CHART_LABELS dict keys (ANN variants - keys are exp_key identifiers)
    ('"ANN-Exp2-SE1": "ANN-E2-SE1", "ANN-Exp2-SE2": "ANN-E2-SE2"',
     '"ANN-Exp2-Sub1": "ANN-E2-SE1", "ANN-Exp2-Sub2": "ANN-E2-SE2"'),

    # FEATURE_DESCRIPTIONS dict keys
    ('    "ANN-Exp2-SE1": {',
     '    "ANN-Exp2-Sub1": {'),
    ('    "ANN-Exp2-SE2": {',
     '    "ANN-Exp2-Sub2": {'),

    # _exp_key function logic (exp_key identifiers used in code conditions)
    ('if is_fs and key in ("Exp1", "Exp2-SE1", "Exp2-SE2"):',
     'if is_fs and key in ("Exp1", "Exp2-Sub1", "Exp2-Sub2"):'),

    # ANN comparison table - list of exp_key filters
    ('    keys = ["ANN-Exp1", "ANN-Exp2-SE1", "ANN-Exp2-SE2", "Phase9-ANN"]',
     '    keys = ["ANN-Exp1", "ANN-Exp2-Sub1", "ANN-Exp2-Sub2", "Phase9-ANN"]'),

    # ANN comparison table - dict keys for column labels
    ('        "ANN-Exp2-SE1": "Exp2-S1 (15 feat, ~924/740 rows)",',
     '        "ANN-Exp2-Sub1": "Exp2-S1 (15 feat, ~924/740 rows)",'),
    ('        "ANN-Exp2-SE2": "Exp2-S2 (19 feat, ~920/733 rows)",',
     '        "ANN-Exp2-Sub2": "Exp2-S2 (19 feat, ~920/733 rows)",'),

    # Phase 9 section - exp_key arguments to subsection builders
    ('        df_all, "ANN-Exp2-SE1", "p9-ann-exp2s1",',
     '        df_all, "ANN-Exp2-Sub1", "p9-ann-exp2s1",'),
    ('        df_all, "ANN-Exp2-SE2", "p9-ann-exp2s2",',
     '        df_all, "ANN-Exp2-Sub2", "p9-ann-exp2s2",'),

    # ANN chart key filter
    ('        for k in ["ANN-Exp1", "ANN-Exp2-SE1", "ANN-Exp2-SE2"]',
     '        for k in ["ANN-Exp1", "ANN-Exp2-Sub1", "ANN-Exp2-Sub2"]'),
]

changed = 0
for broken, correct in REVERTS:
    if broken in content:
        content = content.replace(broken, correct, 1)
        changed += 1
        print(f"  REVERTED: {broken[:60]!r}")
    else:
        print(f"  NOT FOUND: {broken[:60]!r}")

with open(TARGET, "w") as f:
    f.write(content)

print(f"\nDone. {changed} reverts applied.")
