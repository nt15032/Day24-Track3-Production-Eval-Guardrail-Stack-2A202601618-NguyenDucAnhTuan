"""
Setup script: chạy Day 18 pipeline trên 50 câu hỏi → lưu answers_50q.json

Chạy TRƯỚC khi bắt đầu Phase A:
    python setup_answers.py

Yêu cầu:
    1. Đã copy src/ từ Day 18 (m1-m5, pipeline.py) vào thư mục này
    2. docker compose up -d  (Qdrant đang chạy trên port 6333)
    3. .env có OPENAI_API_KEY
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import unicodedata
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def check_day18_files() -> bool:
    required = [
        "src/m1_chunking.py", "src/m2_search.py", "src/m3_rerank.py",
        "src/m4_eval.py",     "src/m5_enrichment.py", "src/pipeline.py",
    ]
    missing = [f for f in required if not os.path.exists(f)]
    if missing:
        print("\n❌ Thiếu files từ Day 18. Copy chúng vào src/ trước:\n")
        for f in missing:
            print(f"   cp <Day18>/src/{os.path.basename(f)} src/")
        return False
    print(f"✓ Day 18 source files: {len(required)}/{len(required)} found")
    return True


def build_pipeline():
    from src.m1_chunking import load_documents, chunk_hierarchical
    from src.m2_search import HybridSearch
    from src.m3_rerank import CrossEncoderReranker
    from src.m5_enrichment import enrich_chunks
    from config import RERANK_TOP_K

    print("\n[1/3] Chunking + enriching documents...")
    t0 = time.time()
    docs = load_documents()
    all_chunks = []
    for doc in docs:
        parents, children = chunk_hierarchical(doc["text"], metadata=doc["metadata"])
        for child in children:
            all_chunks.append({
                "text": child.text,
                "metadata": {**child.metadata, "parent_id": child.parent_id},
            })

    enriched = enrich_chunks(all_chunks)
    if enriched:
        all_chunks = [{"text": e.enriched_text, "metadata": e.auto_metadata} for e in enriched]
        print(f"  ✓ Enriched {len(enriched)} chunks ({time.time()-t0:.1f}s)")
    else:
        print(f"  ✓ Using {len(all_chunks)} raw chunks (M5 not implemented or no API key)")

    print("\n[2/3] Loading reranker...")
    t0 = time.time()
    reranker = CrossEncoderReranker()
    reranker._load_model()  # load before the dense encoder — loading two different
    # safetensors models in the opposite order segfaults on Windows (mmap conflict).
    print(f"  ✓ Reranker ready ({time.time()-t0:.1f}s)")

    print("\n[3/3] Indexing (BM25 + Dense)...")
    t0 = time.time()
    search = HybridSearch()
    search.index(all_chunks)
    print(f"  ✓ Indexed {len(all_chunks)} chunks ({time.time()-t0:.1f}s)")

    return search, reranker, RERANK_TOP_K


@dataclass
class _LocalResult:
    text: str
    score: float
    metadata: dict


def _normalize(text: str) -> set[str]:
    normalized = unicodedata.normalize("NFD", text.casefold())
    normalized = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    return set(re.findall(r"[a-z0-9]+", normalized))


class _LocalSearch:
    def __init__(self, chunks: list[dict]):
        self.chunks = chunks

    def search(self, query: str, top_k: int = 20):
        query_tokens = _normalize(query)
        scored = []
        for chunk in self.chunks:
            tokens = _normalize(chunk["text"])
            overlap = len(query_tokens & tokens) / len(query_tokens) if query_tokens else 0.0
            # Current-policy markers break ties in version-conflict questions.
            current_boost = 0.08 if any(marker in chunk["text"].casefold()
                                        for marker in ["hiện hành", "thay thế hoàn toàn", "phiên bản: 2.0"]) else 0.0
            scored.append(_LocalResult(chunk["text"], overlap + current_boost,
                                       chunk.get("metadata", {})))
        return sorted(scored, key=lambda result: result.score, reverse=True)[:top_k]


class _LocalReranker:
    def rerank(self, query: str, documents: list[dict], top_k: int = 3):
        query_tokens = _normalize(query)
        results = []
        for document in documents:
            tokens = _normalize(document["text"])
            coverage = len(query_tokens & tokens) / len(query_tokens) if query_tokens else 0.0
            results.append(_LocalResult(document["text"],
                                        0.6 * document.get("score", 0.0) + 0.4 * coverage,
                                        document.get("metadata", {})))
        return sorted(results, key=lambda result: result.score, reverse=True)[:top_k]


def build_offline_pipeline():
    """Local lexical fallback when Qdrant/model dependencies are unavailable."""
    from src.m1_chunking import chunk_hierarchical, load_documents
    from config import RERANK_TOP_K

    chunks = []
    for document in load_documents():
        _, children = chunk_hierarchical(document["text"], metadata=document["metadata"])
        chunks.extend({"text": child.text, "metadata": child.metadata} for child in children)
    print(f"  ✓ Offline fallback indexed {len(chunks)} chunks")
    return _LocalSearch(chunks), _LocalReranker(), RERANK_TOP_K


def run_query(q: str, search, reranker, top_k: int) -> tuple[str, list[str]]:
    from config import OPENAI_API_KEY

    results = search.search(q)
    docs    = [{"text": r.text, "score": r.score, "metadata": r.metadata} for r in results]
    reranked = reranker.rerank(q, docs, top_k=top_k)
    contexts = [r.text for r in reranked] if reranked else [r.text for r in results[:3]]

    if OPENAI_API_KEY and contexts:
        try:
            from openai import OpenAI
            client = OpenAI()
            ctx = "\n\n".join(contexts)
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Trả lời CHỈ dựa trên context. Nếu không có → nói 'Không tìm thấy.'"},
                    {"role": "user",   "content": f"Context:\n{ctx}\n\nCâu hỏi: {q}"},
                ],
            )
            return resp.choices[0].message.content, contexts
        except Exception as e:
            print(f"  ⚠️  LLM generation failed: {e}")

    return (contexts[0] if contexts else "Không tìm thấy thông tin."), contexts


def main():
    print("=" * 60)
    print("LAB 24 SETUP — Generating answers for 50 questions")
    print("=" * 60)

    if not check_day18_files():
        sys.exit(1)

    with open("test_set_50q.json", encoding="utf-8") as f:
        test_set = json.load(f)
    print(f"✓ Loaded {len(test_set)} questions (factual/multi_hop/adversarial)")

    try:
        search, reranker, top_k = build_pipeline()
    except Exception as e:
        print(f"\n⚠️  Production pipeline unavailable ({type(e).__name__}: {e})")
        print("→ Dùng local lexical fallback để tạo artifact CI reproducible.")
        search, reranker, top_k = build_offline_pipeline()

    print(f"\nRunning {len(test_set)} queries...")
    answers = []
    t_start = time.time()

    for i, item in enumerate(test_set):
        answer, contexts = run_query(item["question"], search, reranker, top_k)
        answers.append({
            "id":           item["id"],
            "distribution": item["distribution"],
            "question":     item["question"],
            "answer":       answer,
            "contexts":     contexts,
            "ground_truth": item["ground_truth"],
        })
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(test_set)}] done ({time.time()-t_start:.0f}s elapsed)")

    with open("answers_50q.json", "w", encoding="utf-8") as f:
        json.dump(answers, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Saved {len(answers)} answers → answers_50q.json")
    print(f"  Total time: {time.time()-t_start:.1f}s")
    print("\n→ Bây giờ bắt đầu Phase A:")
    print("     python src/phase_a_ragas.py")


if __name__ == "__main__":
    main()
