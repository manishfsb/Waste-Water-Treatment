import os
import json
import csv
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "2020", "RAW.csv")
JSON_2021_PATH = os.path.join(BASE_DIR, "2021", "data", "all_months.json")

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
            pc_kw  = pc_mwh * 1000 if pc_mwh is not None else None

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
    
    # Load existing 2021 JSON
    if os.path.exists(JSON_2021_PATH):
        with open(JSON_2021_PATH, "r", encoding="utf-8") as f:
            year_data = json.load(f)
    else:
        year_data = {}

    count = 0
    for r in rows:
        if r["date"].startswith("2021-"):
            m = r["date"].split("-")[1]
            month_num = int(m)
            # Only process Jan-June (1-6) from CSV
            if month_num <= 6:
                month_key = datetime.strptime(m, "%m").strftime("%B").lower()
                if month_key not in year_data:
                    year_data[month_key] = {
                        "month": month_key.capitalize(),
                        "year": 2021,
                        "control_limits": {},
                        "days": []
                    }
                # Check if date already exists to avoid duplicates
                existing_dates = [d["date"] for d in year_data[month_key]["days"]]
                if r["date"] not in existing_dates:
                    year_data[month_key]["days"].append(r)
                    count += 1
            
    with open(JSON_2021_PATH, "w", encoding="utf-8") as f:
        json.dump(year_data, f, indent=4)
        print(f"Added {count} new days for Jan-June 2021 into {JSON_2021_PATH}")

if __name__ == "__main__":
    main()
