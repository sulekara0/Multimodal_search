# app.py — Streamlit arayüzü (CLIP + FAISS)
import os, json, time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import streamlit as st
from PIL import Image

import torch
from transformers import CLIPModel, CLIPProcessor

import faiss  # pip install faiss-cpu
from deep_translator import GoogleTranslator

st.set_page_config(page_title="Multimodal Search", layout="wide")

# ==== Paths / Config ====
IMG_DIR  = Path(os.environ.get("IMG_DIR", "data/images"))
ART_DIR  = Path(os.environ.get("ART_DIR", "artifacts"))
MODEL_ID = os.environ.get("CLIP_MODEL", "openai/clip-vit-base-patch32")

IDS_PATH   = ART_DIR / "img_ids.json"
INDEX_BIN  = ART_DIR / "faiss_index.bin"
INDEX_META = ART_DIR / "faiss_meta.json"

def maybe_normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / n

@st.cache_resource(show_spinner=False)
def load_clip(model_id: str = MODEL_ID):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(model_id).to(device).eval()
    proc  = CLIPProcessor.from_pretrained(model_id)
    return device, model, proc

@st.cache_resource(show_spinner=False)
def load_index_and_ids():
    if not INDEX_BIN.exists() or not INDEX_META.exists():
        st.error("Kayıtlı FAISS index bulunamadı. Önce: `python scripts\\faiss_search.py build --metric ip`")
        st.stop()
    index = faiss.read_index(str(INDEX_BIN))
    try:
        meta  = json.load(open(INDEX_META, "r", encoding="utf-8"))
    except Exception:
        meta  = json.load(open(INDEX_META, "r", encoding="utf-8-sig"))
    metric = meta.get("metric", "ip")   # "ip" (cosine) veya "l2"

    try:
        paths = json.load(open(IDS_PATH, "r", encoding="utf-8"))
    except Exception:
        paths = json.load(open(IDS_PATH, "r", encoding="utf-8-sig"))
    return index, metric, paths

def resolve_path(image_id: str) -> Path:
    p = Path(image_id)
    if p.exists():
        return p
    return IMG_DIR / image_id

def translate_to_english(text: str, source_lang: str) -> Tuple[str, str]:
    """Seçilen dile göre çeviri yap. Returns (translated_text, source_lang)"""
    if source_lang == 'tr':
        try:
            translated = GoogleTranslator(source='tr', target='en').translate(text)
            return translated, 'tr'
        except Exception as e:
            st.warning(f"Çeviri hatası: {e}. Orijinal metin kullanılıyor.")
            return text, 'tr'
    else:
        # İngilizce seçiliyse direkt kullan
        return text, 'en'

def encode_text_cached(text: str, device, model, proc) -> np.ndarray:
    """Model ile text encoding yap (cache dışında)"""
    with torch.inference_mode():
        t = proc(text=[text], return_tensors="pt", padding=True, truncation=True)
        t = {k: v.to(device) for k, v in t.items()}
        feats = model.get_text_features(**t).detach().cpu().numpy().astype("float32")
    return feats

def search(index, metric: str, ids: List[str], q: str, device, model, proc, k: int = 12):
    qv = encode_text_cached(q, device, model, proc)  # model parametrelerini geç
    if metric == "ip":      # cosine
        qv = maybe_normalize(qv)
        scores, I = index.search(qv, k)         # büyük = iyi
        sims = scores[0].tolist()
    else:                   # l2
        dists, I = index.search(qv, k)          # küçük = iyi -> gösterim için eksiye çevir
        sims = (-dists[0]).tolist()
    idxs = I[0].tolist()
    return [(ids[i], float(sims[j])) for j, i in enumerate(idxs)]

# ==== UI ====
st.title("🔎 Multimodal Görsel Arama (CLIP + FAISS)")

device, model, proc = load_clip(MODEL_ID)
index, metric, paths = load_index_and_ids()

# Dil seçimi toggle
col_lang, col_space = st.columns([1, 5])
with col_lang:
    lang_option = st.toggle("🇹🇷 Türkçe", value=False, help="Açık: Türkçe → İngilizce çeviri | Kapalı: Doğrudan İngilizce")
    selected_lang = 'tr' if lang_option else 'en'

col1, col2 = st.columns([4,1])
with col1:
    placeholder_text = "Örn: 'deniz kenarında kırmızı araba'" if selected_lang == 'tr' else "e.g., 'red car by the sea'"
    label_text = "Metin sorgusu gir (Türkçe)" if selected_lang == 'tr' else "Enter text query (English)"
    q = st.text_input(label_text, placeholder=placeholder_text)
with col2:
    topk = st.slider("Top-K", 4, 24, 12, step=4)

if q.strip():
    # Seçilen dile göre çeviri yap
    translated_query, source_lang = translate_to_english(q.strip(), selected_lang)
    
    t0 = time.perf_counter()
    results = search(index, metric, paths, translated_query, device, model, proc, k=topk)
    dt = time.perf_counter() - t0
    
    # Dil bilgisini göster
    lang_icon = "🇹🇷" if source_lang == 'tr' else "🇬🇧"
    translation_info = f"{lang_icon} {source_lang.upper()}"
    if source_lang == 'tr':
        translation_info += f" → EN: '{translated_query}'"
    
    st.caption(f"Latency: **{dt*1000:.1f} ms** • metric: **{metric}** • Top-K: **{topk}** • {translation_info}")

    cols = st.columns(4)
    for i, (p, score) in enumerate(results):
        with cols[i % 4]:
            path = resolve_path(p)
            try:
                st.image(Image.open(path).convert("RGB"), caption=f"{path.name} • {score:.3f}", use_container_width=True)
            except Exception:
                st.warning(f"Açılamadı: {path}")
