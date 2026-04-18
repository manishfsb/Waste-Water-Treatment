import os
import json
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
YEARS = [2020, 2021, 2022, 2023, 2024, 2025]

def get_data():
    all_rows = []
    for year in YEARS:
        path = os.path.join(BASE_DIR, str(year), "data", "all_months.json")
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            year_data = json.load(f)
            for month, data in year_data.items():
                all_rows.extend(data.get("days", []))
    return all_rows

def main():
    rows = get_data()
    df = pd.DataFrame(rows)
    print(f"Total rows: {len(df)}")
    
    cols = [
        'flow', 'power_total', 
        'inlet_bod', 'inlet_bod_comp', 
        'inlet_cod', 'inlet_cod_comp',
        'effluent_bod', 'effluent_bod_comp',
        'effluent_cod', 'effluent_cod_comp'
    ]
    
    print("\nMissing Values Count & Percentage:")
    for c in cols:
        if c in df.columns:
            missing = df[c].isna().sum()
            pct = (missing / len(df)) * 100
            print(f"{c:20}: {missing:4} missing ({pct:3.1f}%)")

if __name__ == "__main__":
    main()
