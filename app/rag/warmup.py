"""Run at Docker build time: downloads the embedding model and caches passage vectors, so the first user request is fast.
If it fails the app still works with keyword search only (the /plan/status endpoint reports which mode is active)."""
from retriever import Retriever
r = Retriever(); print("retriever mode:", r.mode, "| passages:", len(r.docs))
