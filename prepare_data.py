import streamlit as st
import json, numpy as np, faiss
from sentence_transformers import SentenceTransformer
from pyserini.search.lucene import LuceneSearcher

# UI setup
st.set_page_config(page_title="Hybrid Search Engine", page_icon="💻", layout="wide")
st.title("💻 Hybrid Semantic Search Engine (BM25 + FAISS)")
alpha = st.sidebar.slider("Fusion weight (Dense vs Sparse)", 0.0, 1.0, 0.5)
top_k = st.sidebar.number_input("Top K results", 1, 20, 5)

# Load BM25
searcher = LuceneSearcher("bm25_index")

# Load FAISS
index = faiss.read_index("faiss_msmarco.index")
with open("faiss_ids.txt", "r", encoding="utf-8") as f:
    faiss_ids = [line.strip() for line in f]

# Map IDs to text
id_to_text = {}
with open("collection_json/msmarco_subset.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        id_to_text[obj["id"]] = obj["contents"]

# Dense model
model = SentenceTransformer("all-MiniLM-L6-v2")

query = st.text_input("Enter your query", placeholder="e.g., What is machine learning?")
if query:
    # BM25
    bm25_hits = searcher.search(query, k=top_k*5)
    bm25_ids = [h.docid for h in bm25_hits]
    bm25_scores = np.array([h.score for h in bm25_hits], dtype=np.float32)

    # Dense
    q_vec = model.encode([query], normalize_embeddings=True)
    sims, idxs = index.search(np.array(q_vec, dtype="float32"), top_k*5)
    dense_ids = [faiss_ids[i] for i in idxs[0]]
    dense_scores = sims[0]

    # Normalize
    def norm(x):
        if len(x) == 0: return x
        m, M = np.min(x), np.max(x)
        return (x - m) / (M - m + 1e-8)

    score_map = {}
    for did, s in zip(bm25_ids, norm(bm25_scores)):
        score_map.setdefault(did, [0.0, 0.0])[0] = float(s)
    for did, s in zip(dense_ids, norm(dense_scores)):
        score_map.setdefault(did, [0.0, 0.0])[1] = float(s)

    fused = []
    for did, (sparse_s, dense_s) in score_map.items():
        hybrid = alpha * dense_s + (1 - alpha) * sparse_s
        fused.append((did, hybrid, sparse_s, dense_s))

    fused.sort(key=lambda x: x[1], reverse=True)
    fused = fused[:top_k]

    st.subheader(f"Top {top_k} results")
    for rank, (did, hybrid, sparse_s, dense_s) in enumerate(fused, start=1):
        text = id_to_text.get(did, "(missing)")
        
    st.subheader(f"Top {top_k} results")
    for rank, (did, hybrid, sparse_s, dense_s) in enumerate(fused, start=1):
        text = id_to_text.get(did, "(missing)")
        st.markdown(
            f"""        
            <div style="background:#f7f9fc;border:1px solid #e6ecf3;border-radius:10px;padding:12px;margin-bottom:10px">
              <div><b>{rank}. Document ID: {did}</b></div>
              <div style="color:#334">📝 {text}</div>
              <div style="margin-top:6px;">
                <span style="background:#eaf2ff;color:#1f4ed8;padding:3px 8px;border-radius:16px;margin-right:6px;">Hybrid: {hybrid:.3f}</span>
                <span style="background:#f1f5f9;color:#475569;padding:3px 8px;border-radius:16px;margin-right:6px;">BM25: {sparse_s:.3f}</span>
                <span style="background:#f1f5f9;color:#475569;padding:3px 8px;border-radius:16px;">Dense: {dense_s:.3f}</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )
