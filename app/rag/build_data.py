"""One-time: convert the FNDDS 2017-18 Excel files into compact CSVs the checker loads."""
import csv, sys, openpyxl
src = sys.argv[1]; out = sys.argv[2]
def rows(path, sheet):
    ws = openpyxl.load_workbook(path, read_only=True)[sheet]
    r = list(ws.iter_rows(values_only=True))
    hdr = [str(c).replace("\n", " ").strip() if c else "" for c in r[1]]
    return hdr, r[2:]
hdr, data = rows(src + "/2017-2018 FNDDS At A Glance - FNDDS Nutrient Values.xlsx", "FNDDS Nutrient Values")
cols = {"Food code":"code","Main food description":"desc","WWEIA Category description":"category",
        "Energy (kcal)":"kcal","Protein (g)":"protein_g","Carbohydrate (g)":"carb_g",
        "Fiber, total dietary (g)":"fiber_g","Total Fat (g)":"fat_g",
        "Fatty acids, total saturated (g)":"satfat_g","Potassium (mg)":"potassium_mg","Sodium (mg)":"sodium_mg"}
ix = {k: hdr.index(k) for k in cols}
n = 0
with open(out + "/fndds_2017_18_per100g.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(cols.values())
    for r in data:
        if r[0] is None: continue
        w.writerow([r[ix[k]] for k in cols]); n += 1
print("nutrient rows", n)
hdr, data = rows(src + "/2017-2018 FNDDS At A Glance - Portions and Weights.xlsx", "Portions and Weights")
print(hdr)
wi = [i for i, h in enumerate(hdr) if "weight" in h.lower()]
pi = hdr.index("Portion description")
print("weight col idx", wi, [hdr[i] for i in wi])
m = 0
with open(out + "/fndds_2017_18_portions.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["code", "portion", "grams"])
    for r in data:
        if r[0] is None or r[pi] is None or "not specified" in str(r[pi]).lower(): continue
        g = r[wi[0]]
        if g in (None, ""): continue
        w.writerow([r[0], r[pi], g]); m += 1
print("portion rows", m)
