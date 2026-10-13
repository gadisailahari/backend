# RAG + checker modules (first build)

| File | What it does | Tested |
|---|---|---|
| safety.py | CKD-EPI 2021 eGFR, kidney-safe mode (eGFR<60 or unknown), nutrient targets, medication-advice and potassium text screen | unit tests (test_core.py) |
| checker.py | Recomputes plan nutrients from USDA FNDDS 2017-18; rejects broken targets, unknown foods, implausible amounts | unit tests (test_core.py) |
| retriever.py | BM25 + optional bge-small embeddings, tag filters, pinned cited key facts | BM25 only (test_facts.py); embeddings NOT tested |
| data/corpus.jsonl | 3,796 passages from 28 sources (tags: source, year, region, role, ocr flag) | |
| data/key_facts.json | 6 cited numeric facts, each matched against corpus text | |
| build_data.py / build_corpus.py | rebuild the data files from the source documents | |

Not built yet: LLM call, /chat and /feedback endpoints, review page, rating form, FastAPI wiring.
Placeholders needing expert sign-off: CKD_HIGH_K_MG_PER_SERVING (safety.py) is NOT from a guideline.
