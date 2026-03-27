# Lab Report Headers - July 2021

This document explains every column header found in `Lab Report JULY-2021.xlsx`.

The spreadsheet tracks a **wastewater treatment plant** in Kathmandu, Nepal (Project 10M170). Data flows from dirty water coming in (**Influent**) through multiple cleaning stages until it's safe to discharge (**Effluent**). Think of it like a multi-step laundry process for water - each stage removes more contaminants.

---

## How to Read This Table

| Symbol | Meaning |
|--------|---------|
| 🟢 | Lower is better (you want less of this in the water) |
| 🔵 | Must stay within a target range |
| 🟡 | Higher is better |
| ⚪ | Operational metric (no simple "better/worse" direction) |

---

## Section 1 - Power Generation & Consumption

These columns track how much energy the plant uses and generates (the plant captures methane gas from decomposing sludge and converts it to electricity).

| Column | Header | Unit | Definition |
|--------|--------|------|------------|
| B | Power Generation from Gas Engine 🟡 | MW | Electricity produced by burning biogas from the digesters. *Think of this as the plant recycling its waste into power.* Higher means more energy self-sufficiency. |
| C | Daily Power Consumption from NEA 🟢 | MW | Electricity purchased from Nepal Electricity Authority (the grid). Lower means the plant relies less on external power. |
| D | Total Power Consumption from NEA & GE ⚪ | MW | Sum of grid power + self-generated power = total energy used that day. |
| E | Total Power (KWh) / Flow (ML) 🟢 | KWh/ML | Energy efficiency - how many kilowatt-hours it takes to treat one million liters of sewage. Control limit: < 482.02 KWh/ML. Lower is more efficient. |

---

## Section 2 - Inlet (Raw Sewage / Influent)

This is the **untreated wastewater** arriving at the plant - the dirtiest the water will ever be. These measurements characterize what's coming in so operators know what they're dealing with.

| Column | Header | Unit | Control Limit | Definition |
|--------|--------|------|---------------|------------|
| F | Raw Sewage Flow ⚪ | MLD (Million Liters per Day) | 32.4 | Volume of sewage entering the plant each day. |
| G | pH 🔵 | - | 6.0–9.0 | Acidity/alkalinity of the incoming water. 7 is neutral (like pure water), below 7 is acidic (like vinegar), above 7 is basic (like baking soda). Sewage plants need it in the 6–9 range so the biological treatment bugs can survive. |
| H | BOD₅ (Biochemical Oxygen Demand, 5-day) 🟢 | mg/L | 300 | Measures how much oxygen microorganisms need to break down organic matter in 5 days. *Analogy: imagine a fish tank - high BOD means the organic waste is consuming so much oxygen that fish would suffocate.* A higher number means dirtier water. In influent, values up to 300 mg/L are expected. In effluent, you want this as low as possible (< 10). |
| I | COD (Chemical Oxygen Demand) 🟢 | mg/L | 800 | Similar to BOD but measures **all** oxidizable matter (not just what microbes eat), so it's always higher than BOD. *Think of BOD as "food for bugs" and COD as "everything a chemical reaction can burn."* High COD = highly polluted. |
| J | TSS (Total Suspended Solids) 🟢 | mg/L | 400 | Weight of tiny particles floating in the water (dirt, organic bits, microbes). *Imagine holding a glass of muddy water up to light - TSS is a measure of that cloudiness.* Lower is cleaner. |
| K | TKN (Total Kjeldahl Nitrogen) 🟢 | mg/L | 79 | Total nitrogen from ammonia + organic sources. Too much nitrogen in discharged water causes algae blooms that choke rivers. |
| L | O & G (Oil & Grease) 🟢 | mg/L | 120 | Fats, oils, and grease floating in the water. These clog pipes and harm biological treatment. *Like pouring cooking oil down your sink - it doesn't dissolve and causes blockages.* |
| M | PO₄ (Phosphate) 🟢 | mg/L | 12 | A nutrient that, like nitrogen, fuels algae overgrowth if too much is discharged. |
| N | Total Coliform 🟢 | mg/L | 10⁷–10⁹ | Count of coliform bacteria (a broad group including harmless and harmful species). Used as an indicator of fecal contamination. In raw sewage, high counts are expected. |
| O | Fecal Coliform 🟢 | mg/L | 10⁶–10⁷ | A subset of total coliforms specifically from fecal matter (includes *E. coli*). More directly indicates sewage contamination. |

---

## Section 3 - Grit Classifier Outlet

The grit classifier removes sand, gravel, and heavy particles from the raw sewage (like sifting rocks out of soil).

| Column | Header | Unit | Definition |
|--------|--------|------|------------|
| Q | TSS 🟢 | mg/L | Total Suspended Solids after grit removal. Should be lower than inlet TSS since heavy particles have been removed. |

---

## Section 4 - Primary Clarifier (Primary Sedimentation Tank)

A large settling tank where gravity pulls heavier solids to the bottom. *Think of it like letting a jar of muddy water sit still - the dirt slowly sinks.*

| Column | Header | Unit | Definition |
|--------|--------|------|------------|
| R | pH 🔵 | - | Acidity of water in the primary clarifier. |
| S | TSS 🟢 | mg/L | Suspended solids after primary settling. Should be noticeably lower than inlet. |
| T | BOD₅ 🟢 | mg/L | Organic load remaining after primary treatment. |
| U | COD 🟢 | mg/L | Chemical oxygen demand after primary treatment. |
| V | PST Sludge Totalizer ⚪ | m³ | Volume of sludge (the settled gunk) pumped out from the bottom of the tank. |

---

## Section 5 - Secondary Clarifier

After biological treatment (aeration), this tank settles out the biological floc (clumps of microbes that ate the pollutants).

| Column | Header | Unit | Definition |
|--------|--------|------|------------|
| W | pH 🔵 | - | |
| X | TSS 🟢 | mg/L | Solids remaining. Should be much lower now. |
| Y | BOD₅ 🟢 | mg/L | Organic load - getting close to discharge quality. |
| Z | COD 🟢 | mg/L | Chemical demand - should be dropping significantly. |
| AA | Existing RAS (Return Activated Sludge) ⚪ | mg/L | Concentration of sludge recycled back to the aeration tank. *The plant sends some "trained" microbes back to keep eating new pollutants - like reusing sourdough starter.* |

---

## Section 6 - Secondary Sedimentation Tank Outlet

The water leaving the secondary sedimentation stage, nearing final treatment.

| Column | Header | Unit | Definition |
|--------|--------|------|------------|
| AB | pH 🔵 | - | |
| AC | TSS 🟢 | mg/L | |
| AD | BOD₅ 🟢 | mg/L | |
| AE | COD 🟢 | mg/L | |
| AF | New RAS ⚪ | mg/L | Return Activated Sludge concentration from the newer system. |

---

## Section 7 - Chlorine Contact Tank (Effluent / Discharge Quality)

This is the **final treatment stage** - chlorine kills remaining pathogens before discharge. These numbers represent the **effluent** (outgoing water) and must meet strict limits.

| Column | Header | Unit | Control Limit | Definition |
|--------|--------|------|---------------|------------|
| AH | pH 🔵 | - | 6.5–8.0 | Must be near-neutral for safe discharge. |
| AI | BOD₅ 🟢 | mg/L | < 10 | Must be drastically lower than inlet. If inlet was 300 and effluent is < 10, the plant removed > 96% of organic pollution. |
| AJ | COD 🟢 | mg/L | < 250 | Chemical demand in the final water. |
| AK | TSS 🟢 | mg/L | < 10 | Water should be almost particle-free. |
| AL | O & G 🟢 | mg/L | < 10 | Virtually no oil or grease remaining. |
| AM | Ammoniacal Nitrogen 🟢 | mg/L | < 50 | Ammonia (NH₃) remaining. Toxic to aquatic life at high levels. |
| AN | Total Coliform 🟢 | MPN/100mL | < 500 | MPN = Most Probable Number. Bacterial count must be orders of magnitude lower than inlet. |
| AO | Fecal Coliform 🟢 | MPN/100mL | < 100 | Critical public health metric. Below 100 means the chlorination is working. |
| AP | FRC (Free Residual Chlorine) 🔵 | mg/L | Max 1 | Chlorine left after disinfection. Needs to be > 0 (to prove disinfection happened) but ≤ 1 mg/L (too much chlorine is toxic to river life). |

---

## Section 8 - Aeration Tanks (Biological Treatment)

This is where the magic happens: billions of microorganisms eat the dissolved pollutants. Air is pumped in to keep these bugs alive and active. The plant has two tanks: **Existing** and **New**.

| Column | Header | Unit | Control Limit | Definition |
|--------|--------|------|---------------|------------|
| AR / AX | pH 🔵 | - | - | Kept stable to keep microbes healthy. |
| AS / AY | DO (Dissolved Oxygen) 🟡 | mg/L | > 0.5 | Oxygen dissolved in the water. Must stay above 0.5 mg/L or the bugs die. *Like making sure there's enough air for workers in a mine.* Higher is better (within reason). |
| AT / AZ | MLSS (Mixed Liquor Suspended Solids) ⚪ | mg/L | - | Concentration of microorganisms + suspended matter in the aeration tank. *This number tells you how many "workers" (bacteria) you have on the job.* Typical range: 2,000–4,000 mg/L. |
| AU / BA | MLVSS (Mixed Liquor Volatile Suspended Solids) ⚪ | mg/L | - | The *living* portion of MLSS (organic/biological solids only, excluding inert grit). Usually 70–85% of MLSS. *If MLSS counts all workers, MLVSS counts only the ones actually doing the work.* |
| AV / BB | SV30 (Sludge Volume after 30 min) ⚪ | mL/L | - | Volume of sludge that settles in a 1-liter cylinder after 30 minutes. *A quick field test: fill a graduated cylinder, wait 30 min, read how much sludge settled.* Indicates settleability. |
| AW / BC | SVI (Sludge Volume Index) 🟢 | - | - | SV30 ÷ MLSS × 1000. Measures how well sludge compacts. **< 100** = good settling, **100–200** = moderate, **> 200** = bulking (fluffy sludge that won't settle - a plant operator's nightmare). Lower is better. |

> **Note**: Columns AR–AW are the **Existing Aeration Tank**; columns AX–BC are the **New Aeration Tank**. Same parameters, different equipment.

---

## Section 9 - Digester Sludge Feeding Details

Sludge from the clarifiers is fed into anaerobic digesters (sealed tanks with no oxygen) where bacteria break it down and produce biogas. This section tracks what goes *into* the digesters.

| Column | Header | Unit | Definition |
|--------|--------|------|------------|
| **Primary Sedimentation Tank Underflow** | | | *Sludge pumped from the primary clarifier bottom* |
| BE | pH 🔵 | - | |
| BF | TS (Total Solids) ⚪ | % | Percentage of the sludge that is solid material (the rest is water). *If TS = 4%, then for every 100 kg of sludge, 4 kg is actual solid matter.* |
| BG | VS (Volatile Solids) ⚪ | % | The portion of total solids that is organic and can be digested (burned off at 550°C). A higher VS/TS ratio means more "food" for the digester bacteria. |
| **Thickener Sludge Parameters** | | | *Sludge after mechanical thickening (squeezing water out)* |
| BH | pH 🔵 | - | |
| BI | TS ⚪ | % | Should be higher than before thickening (more concentrated). |
| BJ | VS ⚪ | % | |
| BK | VFA (Volatile Fatty Acids) ⚪ | mg/L | Short-chain fatty acids produced during early digestion. *Too many VFAs mean the digester is "overfed" - like giving someone more food than they can handle.* |
| BL | Alkalinity 🟡 | mg/L | The water's ability to neutralize acids (buffering capacity). *Like antacids for the digester - keeps the pH from crashing.* Higher alkalinity = more stable digester. |
| BM | VFA/Alkalinity Ratio 🟢 | - | **Critical stability indicator.** < 0.3 = healthy digester, 0.3–0.5 = watch closely, > 0.5 = digester stress/failure risk. Lower is better. |
| BN | Digester Feed Totalizer ⚪ | m³ | Total volume of sludge fed into the digester that day. |

---

## Section 10 - Digester Performance Details (Digester A & Digester B)

Monitors conditions inside each of the two anaerobic digesters.

| Column (A / B) | Header | Unit | Definition |
|-----------------|--------|------|------------|
| BP / BW | pH 🔵 | - | Ideal range: 6.8–7.4. If pH drops, the digester is turning acidic (souring). |
| BQ / BX | TS (Total Solids) ⚪ | % | Solid content inside the digester. |
| BR / BY | VS (Volatile Solids) ⚪ | % | Organic content being digested. A decreasing VS over time means digestion is working. |
| BS / BZ | VFA 🟢 | mg/L | Should stay low. Rising VFA = acid buildup = trouble. |
| BT / CA | Alkalinity 🟡 | mg/L | Higher = better buffered against pH swings. |
| BU / CB | VFA/Alkalinity Ratio 🟢 | - | Same as above: < 0.3 is healthy. |
| BV / CC | Temperature 🔵 | °C | Digesters operate in a mesophilic range (30–38°C). Too cold = slow digestion, too hot = microbe die-off. |

---

## Section 11 - Digester Outlet & Centrifuge

What comes *out* of the digesters and how the sludge is dewatered.

| Column | Header | Unit | Definition |
|--------|--------|------|------------|
| CE | Digester Underflow pH 🔵 | - | pH of the digested sludge. |
| CF | TS ⚪ | % | Solids content of the digested sludge leaving the digester. |
| CG | VS ⚪ | % | Should be lower than the feed VS (meaning organic matter was consumed). |
| CH | Centrifuge Cake TS 🟡 | % | Total solids in the dewatered sludge "cake" after centrifuging. *Like wringing out a wet towel - you want the cake to be as dry as possible.* Higher = drier cake = easier/cheaper to dispose of. |
| CI | Centrifuge Cake VS ⚪ | % | Volatile solids in the cake. |
| CJ | Centrifuge Feed Totalizer ⚪ | m³ | Volume of digested sludge fed into the centrifuge. |

---

## Section 12 - Biogas

Gas generated by the anaerobic digesters.

| Column | Header | Unit | Definition |
|--------|--------|------|------------|
| CK | Gas Generation 🟡 | Nm³ (Normal cubic meters) | Total biogas produced. Nm³ is volume at standard temperature and pressure. More gas = more energy potential. |
| CL | CH₄ (Methane) 🟡 | % | Percentage of the biogas that is methane - the valuable, combustible part. Typical: 55–70%. Higher methane = better fuel. |
| CM | CO₂ (Carbon Dioxide) 🟢 | % | The non-combustible portion. Lower is better (means more methane). |

---

## Section 13 - Scrubber

Removes hydrogen sulfide (H₂S) from the biogas before it enters the gas engine. H₂S is toxic and corrosive - it smells like rotten eggs.

| Column | Header | Unit | Definition |
|--------|--------|------|------------|
| CN | H₂S Inlet ⚪ | ppm | Concentration of hydrogen sulfide entering the scrubber. |
| CO | H₂S Outlet 🟢 | ppm | Concentration leaving the scrubber. Must be drastically lower than inlet. Lower = scrubber is working well. |

---

## Section 14 - Odour Control Unit

Treats foul air from the plant to prevent bad smells in the surrounding area.

| Column | Header | Unit | Definition |
|--------|--------|------|------------|
| CP | Inlet pH ⚪ | - | pH of the scrubbing liquid entering the odour control system. |
| CQ | Outlet pH ⚪ | - | pH of the scrubbing liquid exiting. The change in pH indicates how much odour-causing compounds were absorbed. |

---

## Quick-Reference: Key Influent vs. Effluent Comparison

These are the parameters you care about most - they show how effective the plant is:

| Parameter | Influent (Inlet) Limit | Effluent (CCT) Limit | What This Means |
|-----------|----------------------|---------------------|-----------------|
| **pH** | 6.0–9.0 | 6.5–8.0 | Range gets tighter for discharge |
| **BOD₅** | 300 mg/L | < 10 mg/L | Must remove > 96% of organic pollution |
| **COD** | 800 mg/L | < 250 mg/L | Must remove > 68% of chemical oxygen demand |
| **TSS** | 400 mg/L | < 10 mg/L | Must remove > 97% of suspended particles |
| **O & G** | 120 mg/L | < 10 mg/L | Must remove > 91% of oils and grease |
| **Total Coliform** | 10⁷–10⁹ | < 500 MPN/100mL | Bacteria reduced by ~99.999% |
| **Fecal Coliform** | 10⁶–10⁷ | < 100 MPN/100mL | Fecal bacteria nearly eliminated |
