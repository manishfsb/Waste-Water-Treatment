import os
import json
import csv
from collections import defaultdict
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "RAW.csv")
OUT_DIR = os.path.join(BASE_DIR, "extracted_data")
OUT_FILE = os.path.join(OUT_DIR, "all_months.json")

def load_csv(csv_path):
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for rec in reader:
            def v(col):
                raw = rec.get(col, "").strip()
                if raw in ("", "-", "N/A"):
                    return None
                try:
                    return float(raw)
                except ValueError:
                    return None

            date_str = rec.get("Date", "").strip()
            try:
                dt = datetime.strptime(date_str, "%m/%d/%Y")
                date_fmt = dt.strftime("%Y-%m-%d")
            except ValueError:
                date_fmt = date_str

            pc_mwh = v("PC")
            pc_kw  = pc_mwh * 1000 if pc_mwh is not None else None   # MWh → KW

            row = {
                "date":              date_fmt,
                "power_ge":          None,
                "power_nea":         pc_kw,
                "power_total":       pc_kw,
                "power_per_flow":    v("PF"),
                "flow":              v("Q_IN"),
                "inlet_ph":          v("IN_pH"),
                "inlet_bod":         v("IN_BOD"),
                "inlet_cod":         v("IN_COD"),
                "inlet_tss":         v("IN_TSS"),
                "inlet_ph_comp":     None,
                "inlet_bod_comp":    None,
                "inlet_cod_comp":    None,
                "inlet_tss_comp":    None,
                "effluent_ph":       None,
                "effluent_bod":      v("O_BOD"),
                "effluent_cod":      v("O_COD"),
                "effluent_tss":      v("O_TSS"),
                "effluent_frc":      None,
                "effluent_ph_comp":  None,
                "effluent_bod_comp": None,
                "effluent_cod_comp": None,
                "effluent_tss_comp": None,
            }
            rows.append(row)
    return rows

def main():
    rows = load_csv(CSV_PATH)
    
    # Filter for year 2020 only
    year_data = {}
    for r in rows:
        if r["date"].startswith("2020-"):
            m = r["date"].split("-")[1]
            month_key = datetime.strptime(m, "%m").strftime("%B").lower()
            if month_key not in year_data:
                year_data[month_key] = {
                    "month": month_key.capitalize(),
                    "year": 2020,
                    "control_limits": {},
                    "days": []
                }
            year_data[month_key]["days"].append(r)
            
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(year_data, f, indent=4)
        print(f"Wrote {sum(len(v['days']) for v in year_data.values())} days for 2020 into {OUT_FILE}")

if __name__ == "__main__":
    main()
