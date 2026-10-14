"""Hybrid retriever over the guideline corpus: BM25 + (optional) open embedding model, fused with
reciprocal-rank fusion. Tags (role/region/source) filter the candidates.

Embeddings use fastembed's BAAI/bge-small-en-v1.5 (runs inside our server, no API key). If fastembed is
not installed or the model cannot download, the retriever silently falls back to BM25 only and says so in
`self.mode`, so results are never presented as embedding-based when they are not.
"""
import json, re, pathlib
from rank_bm25 import BM25Okapi

HERE = pathlib.Path(__file__).parent
_tok = lambda s: re.findall(r"[a-z0-9]+", s.lower())
DEFAULT_EXCLUDE_ROLES = {"comparison"}          # e.g. Australian guidelines: only on request


class Retriever:
    def __init__(self, corpus_path=HERE / "data/corpus.jsonl", use_embeddings="auto"):
        self.docs = [json.loads(l) for l in open(corpus_path)]
        self.bm25 = BM25Okapi([_tok(d["text"] + " " + d["title"]) for d in self.docs])
        self.mode, self.emb, self.vecs = "bm25", None, None
        if use_embeddings in ("auto", True):
            try:
                import numpy as np
                from fastembed import TextEmbedding
                cache = HERE / "data/embeddings.npy"
                self.emb = TextEmbedding("BAAI/bge-small-en-v1.5")
                if cache.exists():
                    self.vecs = np.load(cache)
                else:
                    self.vecs = np.array(list(self.emb.embed([d["text"] for d in self.docs])))
                    np.save(cache, self.vecs)
                self.vecs = self.vecs / np.linalg.norm(self.vecs, axis=1, keepdims=True)
                self.mode = "hybrid(bm25+bge-small)"
            except Exception as e:           # no package / no network: fall back honestly
                self.emb = self.vecs = None
                self.mode = f"bm25 (embeddings unavailable: {type(e).__name__})"
                if use_embeddings is True: raise

    def search(self, query, k=5, roles=None, regions=None, sources=None, include_comparison=False):
        ok = []
        for i, d in enumerate(self.docs):
            if not include_comparison and d["role"] in DEFAULT_EXCLUDE_ROLES: continue
            if roles and d["role"] not in roles: continue
            if regions and d["region"] not in regions: continue
            if sources and d["source"] not in sources: continue
            ok.append(i)
        bm = self.bm25.get_scores(_tok(query))
        ranks = {}
        for r, i in enumerate(sorted(ok, key=lambda i: -bm[i])[:50]): ranks[i] = ranks.get(i, 0) + 1 / (60 + r)
        if self.vecs is not None:
            import numpy as np
            q = np.array(list(self.emb.embed([query])))[0]; q = q / np.linalg.norm(q)
            sims = self.vecs @ q
            for r, i in enumerate(sorted(ok, key=lambda i: -sims[i])[:50]): ranks[i] = ranks.get(i, 0) + 1 / (60 + r)
        top = sorted(ranks, key=lambda i: -ranks[i])[:k]
        return [{**self.docs[i], "score": round(ranks[i], 4)} for i in top]


def plan_facts(structured):
    """The cited facts the plan writer sees (kept to about 8 to save tokens). Chosen by the user's situation."""
    facts = {f["id"]: f for f in json.load(open(HERE / "data/key_facts.json"))}
    ckd = structured.get("kidney_safe_mode")
    ids = ["sodium_ckd" if ckd else "sodium_general", "dash_servings_2000kcal", "exercise_rec", "exercise_table", "whole_grains"]
    if ckd: ids += ["ckd_dietitian", "salt_substitute_ckd"]
    else: ids += ["protein_amdr"]
    conds = set(structured.get("conditions") or [])
    if "pcos" in conds: ids.append("pcos")
    if "cholesterol" in conds: ids.append("satfat")
    if "yoga" in set(structured.get("activities") or []): ids.append("meditation")
    d = (structured.get("diet_type") or "").lower()
    if not ("vegan" in d or ("vegetarian" in d and "non" not in d)): ids.append("fish_twice")
    return [facts[i] for i in ids if i in facts][:9]


def pinned_facts(ckd_mode):
    """Numeric targets that are always shown to the LLM (and used by the checker), each with a real citation."""
    facts = json.load(open(HERE / "data/key_facts.json"))
    keep = {"sodium_ckd", "ckd_definition", "potassium_ckd", "dash_servings_2000kcal"} if ckd_mode else \
           {"sodium_general", "potassium_ai", "dash_servings_2000kcal", "ckd_definition"}
    return [f for f in facts if f["id"] in keep]
