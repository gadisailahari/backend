"""Build data/corpus.jsonl from the extracted guideline text (run once; re-run when sources are added)."""
import json, re, sys, pathlib
S = pathlib.Path(sys.argv[1])            # scratchpad
chunks = [json.loads(l) for l in open(S / "rag_work/chunks_v2.jsonl")]
# add sources extracted separately
extra = [
 (S/"rag_work/aha2021_plain.txt","aha_2021_diet","2021 Dietary Guidance to Improve Cardiovascular Health (AHA Scientific Statement)","AHA",2021,"USA","core",False),
 (S/"rag_work/kdigo_ckd.txt","kdigo_ckd_2024","KDIGO 2024 Clinical Practice Guideline for Evaluation and Management of CKD","KDIGO",2024,"Global","kidney-safety",False),
 (S/"kidney_ocr/ods.txt","nih_ods_potassium","Potassium: Fact Sheet for Health Professionals","NIH Office of Dietary Supplements",None,"USA","kidney-safety",True),
]
for path, sid, title, pub, yr, reg, role, ocr in extra:
    pages = path.read_text().split("\f")
    for pi, pg in enumerate(pages, 1):
        w = re.sub(r"\s+", " ", pg).split(); i = 0
        while i < len(w):
            piece = " ".join(w[i:i+220])
            if len(piece.split()) >= 25:
                chunks.append(dict(id=f"{sid}:p{pi}:{i}", source=sid, title=title, publisher=pub, year=yr, region=reg, role=role, ocr=ocr, page=pi, text=piece))
            i += 180
# some older entries were tagged with the wrong role for kidney docs
NOISE = [r"\d{1,2}/\d{1,2}/\d{2}, \d{1,2}:\d{2} [AP]M", r"We use essential cookies[^.]*\.", r"Manage Cookie Preferences", r"https?://\S+"]
out, dropped = [], 0
for c in chunks:
    t = c["text"]
    for p in NOISE: t = re.sub(p, " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    # drop reference-list chunks (many citations, little prose)
    if len(re.findall(r"\bet al\b", t)) >= 4 or len(re.findall(r"\b(19|20)\d{2};\d", t)) >= 4:
        dropped += 1; continue
    if len(t.split()) < 25: dropped += 1; continue
    c["text"] = t; out.append(c)
pathlib.Path("data/corpus.jsonl").write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in out))
from collections import Counter
print("kept", len(out), "dropped", dropped); print(Counter(c["source"] for c in out).most_common(40))
