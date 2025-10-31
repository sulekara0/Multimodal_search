import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

# -------------------------------------------------------------
# model_embedding.py
#   - CLIP (openai/clip-vit-base-patch32) ile görsel & metin embedding üretimi
#   - Embedding'leri .npy dosyasına ve id'leri .json'a kaydeder
#   - Basit bir cosine similarity araması yapar ("Kırmızı araba" örneği dahil)
#   - Kullanım örnekleri dosyanın en altında
# -------------------------------------------------------------

@torch.inference_mode()
def load_clip(model_name: str = "openai/clip-vit-base-patch32", device: str = None):
    """CLIP model & processor yükler.
    Args:
        model_name: HF model id'si
        device: cuda, mps veya cpu. None ise otomatik seçilir.
    Returns: (model, processor, device)
    """
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"  # Apple Silicon
        else:
            device = "cpu"

    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval().to(device)
    return model, processor, device


def _gather_image_paths(images_dir: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths = [p for p in images_dir.rglob("*") if p.suffix.lower() in exts]
    paths.sort()
    if not paths:
        raise FileNotFoundError(f"{images_dir} içinde desteklenen uzantılarda görsel bulunamadı.")
    return paths


@torch.inference_mode()
def compute_image_embeddings(
    images_dir: Path,
    output_dir: Path,
    model: CLIPModel,
    processor: CLIPProcessor,
    device: str,
    batch_size: int = 32,
) -> Tuple[np.ndarray, List[str]]:
    """Klasördeki tüm görseller için embedding hesaplar ve döner.
    Ayrıca output_dir içine embeddings.npy ve ids.json kayıt edilir.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    img_paths = _gather_image_paths(images_dir)
    ids = [p.name for p in img_paths]

    all_embeds = []
    for i in tqdm(range(0, len(img_paths), batch_size), desc="Embedding (image)"):
        batch_paths = img_paths[i : i + batch_size]
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
        image_features = model.get_image_features(**inputs)
        image_features = torch.nn.functional.normalize(image_features, p=2, dim=1)
        all_embeds.append(image_features.cpu())

    embeds = torch.cat(all_embeds, dim=0).numpy()

    # Kaydet
    np.save(output_dir / "embeddings.npy", embeds)
    with open(output_dir / "ids.json", "w", encoding="utf-8") as f:
        json.dump(ids, f, ensure_ascii=False, indent=2)

    # Basit bir özet
    print(f"[INFO] embeddings shape: {embeds.shape} | ids: {len(ids)}")
    return embeds, ids


@torch.inference_mode()
def encode_texts(texts: List[str], model: CLIPModel, processor: CLIPProcessor, device: str) -> np.ndarray:
    inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(device)
    text_features = model.get_text_features(**inputs)
    text_features = torch.nn.functional.normalize(text_features, p=2, dim=1)
    return text_features.cpu().numpy()


def cosine_topk(query_vec: np.ndarray, matrix: np.ndarray, k: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """query_vec: (D,), matrix: (N, D) -> returns (topk_scores, topk_indices)"""
    # matrix ve query zaten normalize ise doğrudan dot product = cosine
    scores = matrix @ query_vec
    topk_idx = np.argpartition(-scores, kth=min(k, len(scores)-1))[:k]
    topk_idx = topk_idx[np.argsort(-scores[topk_idx])]  # skorlarına göre sırala
    return scores[topk_idx], topk_idx


def search_text(
    query: str,
    embeddings_path: Path,
    ids_path: Path,
    model: CLIPModel,
    processor: CLIPProcessor,
    device: str,
    topk: int = 3,
):
    # Yükle
    matrix = np.load(embeddings_path)
    with open(ids_path, "r", encoding="utf-8") as f:
        ids = json.load(f)

    # Metni encode et
    q_vec = encode_texts([query], model, processor, device)[0]

    # En benzer K görseli bul
    scores, idxs = cosine_topk(q_vec, matrix, k=topk)

    print("\n🔎 Sorgu:", query)
    print("En yakın görseller:")
    for rank, (score, idx) in enumerate(zip(scores, idxs), start=1):
        print(f" {rank:>2}. {ids[idx]} | cosine={float(score):.4f}")


def save_model_card(output_dir: Path, model_name: str, device: str):
    meta = {"model": model_name, "device": device}
    with open(output_dir / "model_info.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def parse_args():
    p = argparse.ArgumentParser(description="CLIP ile görsel ve metin embedding + arama")
    p.add_argument("images_dir", type=str, help="Görsellerin olduğu klasör (alt klasörler dahil)")
    p.add_argument("output_dir", type=str, help="Çıktıların yazılacağı klasör")
    p.add_argument("--model", type=str, default="openai/clip-vit-base-patch32", help="CLIP model adı")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device", type=str, default=None, choices=[None, "cpu", "cuda", "mps"], help="Zorla cihaz seçimi")
    p.add_argument("--only_search", action="store_true", help="Sadece mevcut embedding'lerle arama yap")
    p.add_argument("--query", type=str, default="Kırmızı araba", help="Arama için sorgu metni")
    p.add_argument("--topk", type=int, default=3)
    return p.parse_args()


def main():
    args = parse_args()

    images_dir = Path(args.images_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, processor, device = load_clip(args.model, args.device)
    save_model_card(output_dir, args.model, device)

    emb_path = output_dir / "embeddings.npy"
    ids_path = output_dir / "ids.json"

    if not args.only_search:
        compute_image_embeddings(images_dir, output_dir, model, processor, device, batch_size=args.batch_size)

    # Basit test: metin embedding + arama
    if emb_path.exists() and ids_path.exists():
        search_text(args.query, emb_path, ids_path, model, processor, device, topk=args.topk)
    else:
        print("[WARN] Arama yapabilmek için embeddings.npy ve ids.json dosyaları gerekli.")


if __name__ == "__main__":
    main()
