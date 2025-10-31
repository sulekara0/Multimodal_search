"""
FAISS text→image search utility (Week 3)

What this script does
- Builds a FAISS index (IndexFlatL2 or IndexFlatIP) from precomputed image embeddings
- Adds image embeddings to the index and saves it to disk
- Encodes text queries with CLIP and returns the top‑k nearest images
- Measures performance (latency per query) and simple accuracy metrics if you provide expected matches

Assumptions
- You already created image embeddings with your previous pipeline and saved them to:
    ART_DIR/img_embeddings.npy   (float32 array of shape [N, D])
    ART_DIR/img_ids.json         (list[str] of length N; each id maps to an image filename or unique id)
- Optional: ART_DIR/captions.json for benchmarking (list or dict of {id: caption})
- Images live under IMG_DIR; ids should be resolvable to files (e.g., filename only, or relative path)

Usage examples
# 1) Build / load index
python faiss_search.py build --metric l2           # builds IndexFlatL2 and saves it
python faiss_search.py build --metric ip           # builds IndexFlatIP (cosine if normalized)

# 2) Search with a text query
python faiss_search.py search --query "red car by the sea" --topk 5 --metric ip

# 3) Benchmark latency & quality (requires queries.json)
python faiss_search.py bench --queries artifacts/queries.json --topk 10 --metric ip

# queries.json format examples
[
  {"q": "a tabby cat on sofa", "expected_id": "cat_001.jpg"},
  {"q": "a yellow sports car",  "expected_id": "car_yellow.png"}
]

Notes
- Metric "ip" uses cosine similarity by L2-normalizing vectors (common for CLIP)
- Metric "l2" uses Euclidean distance (as requested example: IndexFlatL2)
- Target: < 0.5s per query on CPU for moderate N (≈1k–10k). For larger N, consider IVF/HNSW or FAISS GPU.
"""

from __future__ import annotations
import os
import json
import time
import argparse
from typing import List, Tuple, Dict

import numpy as np
np.set_printoptions(suppress=True, precision=4)

try:
    import faiss  # faiss-cpu or faiss-gpu
except Exception as e:
    raise SystemExit("[ERROR] FAISS is not installed. Try: pip install faiss-cpu -i https://pypi.org/simple")

import torch
from transformers import CLIPProcessor, CLIPModel

# === Paths (adjust if your project differs) ===
IMG_DIR = os.environ.get("IMG_DIR", "data/images")
ART_DIR = os.environ.get("ART_DIR", "artifacts")
MODEL_ID = os.environ.get("CLIP_MODEL", "openai/clip-vit-base-patch32")

EMB_PATH = os.path.join(ART_DIR, "img_embeddings.npy")
IDS_PATH = os.path.join(ART_DIR, "img_ids.json")
CAPTIONS_PATH = os.path.join(ART_DIR, "captions.json")
INDEX_PATH = os.path.join(ART_DIR, "faiss_index.bin")
META_PATH = os.path.join(ART_DIR, "faiss_meta.json")  # stores metric + dim + n

# === Helpers ===
def load_embeddings() -> Tuple[np.ndarray, List[str]]:
    if not os.path.exists(EMB_PATH):
        raise FileNotFoundError(f"Embeddings not found: {EMB_PATH}")
    if not os.path.exists(IDS_PATH):
        raise FileNotFoundError(f"IDs not found: {IDS_PATH}")

    embs = np.load(EMB_PATH)
    if embs.dtype != np.float32:
        embs = embs.astype(np.float32)
    with open(IDS_PATH, "r", encoding="utf-8") as f:
        ids = json.load(f)
    if len(ids) != embs.shape[0]:
        raise ValueError(f"ids length ({len(ids)}) != embeddings rows ({embs.shape[0]})")
    return embs, ids


def maybe_normalize(x: np.ndarray) -> np.ndarray:
    # L2-normalize along last dim; add eps for safety
    norm = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norm


def build_index(embs: np.ndarray, metric: str = "ip") -> faiss.Index:
    d = embs.shape[1]
    if metric == "l2":
        index = faiss.IndexFlatL2(d)
        index.add(embs)
    elif metric == "ip":
        # Cosine: normalize vectors and use inner product
        embs_norm = maybe_normalize(embs)
        index = faiss.IndexFlatIP(d)
        index.add(embs_norm)
    else:
        raise ValueError("metric must be one of: 'l2', 'ip'")
    return index


def save_index(index: faiss.Index, metric: str, n: int, d: int):
    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump({"metric": metric, "n": n, "d": d}, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Saved index to {INDEX_PATH} and meta to {META_PATH}")


def load_index() -> Tuple[faiss.Index, str, int, int]:
    if not os.path.exists(INDEX_PATH) or not os.path.exists(META_PATH):
        raise FileNotFoundError("Index/meta not found. Run 'build' first.")
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return index, meta["metric"], meta["n"], meta["d"]


# === CLIP encoding ===
class TextEncoder:
    def __init__(self, model_id: str = MODEL_ID, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_id).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.model.eval()
        print(f"[INFO] Using {model_id} on {self.device}")

    @torch.no_grad()
    def encode(self, texts: List[str]) -> np.ndarray:
        inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        out = self.model.get_text_features(**inputs)
        vec = out.detach().cpu().numpy().astype(np.float32)
        return vec


# === Search ===
def search_text(index: faiss.Index, metric: str, ids: List[str], encoder: TextEncoder, query: str, topk: int = 5) -> List[Tuple[str, float]]:
    q = encoder.encode([query])  # (1, D)
    if metric == "ip":
        q = maybe_normalize(q)
        scores, I = index.search(q, topk)  # higher is better
        sims = scores[0].tolist()
    else:  # l2
        dists, I = index.search(q, topk)   # lower is better
        # convert to similarity-like score for display (negate distance)
        sims = (-dists[0]).tolist()

    idxs = I[0].tolist()
    return [(ids[i], sims[j]) for j, i in enumerate(idxs)]


# === Benchmark ===
def benchmark(index: faiss.Index, metric: str, ids: List[str], encoder: TextEncoder, queries: List[Dict], topk: int = 5) -> Dict:
    latencies = []
    hits_at_k = 0
    mrr_sum = 0.0

    for obj in queries:
        q = obj.get("q")
        expected_id = obj.get("expected_id")
        t0 = time.perf_counter()
        results = search_text(index, metric, ids, encoder, q, topk=topk)
        dt = (time.perf_counter() - t0)
        latencies.append(dt)

        # rank metrics
        ranked_ids = [r[0] for r in results]
        if expected_id is not None:
            if expected_id in ranked_ids:
                hits_at_k += 1
                rank = ranked_ids.index(expected_id) + 1
                mrr_sum += 1.0 / rank

    out = {
        "avg_latency_sec": float(np.mean(latencies)) if latencies else None,
        "p95_latency_sec": float(np.percentile(latencies, 95)) if latencies else None,
        "queries": len(queries),
        "topk": topk,
    }
    # only include accuracy metrics if expected_id present in at least one query
    if any(q.get("expected_id") is not None for q in queries):
        total_with_gt = sum(1 for q in queries if q.get("expected_id") is not None)
        out.update({
            "hit_rate@k": hits_at_k / max(1, total_with_gt),
            "mrr": mrr_sum / max(1, total_with_gt),
        })
    return out


# === Utilities ===
def resolve_image_path(image_id: str) -> str:
    # If image_id is already a path, return as-is; else assume under IMG_DIR
    if os.path.exists(image_id):
        return image_id
    return os.path.join(IMG_DIR, image_id)


def pretty_print_results(results: List[Tuple[str, float]]):
    for rank, (iid, score) in enumerate(results, start=1):
        p = resolve_image_path(iid)
        print(f"#{rank:02d}  id={iid}  score={score:.4f}  path={p}")


# === CLI ===
def cmd_build(args):
    embs, ids = load_embeddings()
    print(f"[INFO] Embeddings shape: {embs.shape} | ids: {len(ids)}")
    index = build_index(embs, metric=args.metric)
    save_index(index, metric=args.metric, n=embs.shape[0], d=embs.shape[1])


def cmd_search(args):
    index, metric, n, d = load_index()
    with open(IDS_PATH, "r", encoding="utf-8") as f:
        ids = json.load(f)
    enc = TextEncoder(MODEL_ID)
    results = search_text(index, metric, ids, enc, args.query, topk=args.topk)
    pretty_print_results(results)


def cmd_bench(args):
    index, metric, n, d = load_index()
    with open(IDS_PATH, "r", encoding="utf-8") as f:
        ids = json.load(f)
    enc = TextEncoder(MODEL_ID)

    if not os.path.exists(args.queries):
        raise FileNotFoundError(f"queries file not found: {args.queries}")
    with open(args.queries, "r", encoding="utf-8") as f:
        qlist = json.load(f)
    stats = benchmark(index, metric, ids, enc, qlist, topk=args.topk)

    # Save json report
    report_path = os.path.join(ART_DIR, "faiss_bench_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("\n[RESULTS]")
    for k, v in stats.items():
        print(f"{k}: {v}")
    print(f"\n[INFO] Saved report to {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FAISS text→image search (CLIP)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build", help="Build FAISS index from saved embeddings")
    p_build.add_argument("--metric", choices=["l2", "ip"], default="ip", help="Distance metric")
    p_build.set_defaults(func=cmd_build)

    p_search = sub.add_parser("search", help="Search with a text query")
    p_search.add_argument("--query", required=True, help="Text query to retrieve images")
    p_search.add_argument("--topk", type=int, default=5, help="Top-K results")
    p_search.set_defaults(func=cmd_search)

    p_bench = sub.add_parser("bench", help="Benchmark latency & accuracy")
    p_bench.add_argument("--queries", required=True, help="Path to queries.json")
    p_bench.add_argument("--topk", type=int, default=10)
    p_bench.set_defaults(func=cmd_bench)

    args = parser.parse_args()
    args.func(args)
