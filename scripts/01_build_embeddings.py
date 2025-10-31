import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image, ImageFile
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

# Bozuk/truncated görselleri mümkünse açabilmek için
ImageFile.LOAD_TRUNCATED_IMAGES = True

@torch.inference_mode()
def load_clip(model_name: str = "openai/clip-vit-base-patch32", device: str | None = None):
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    model = CLIPModel.from_pretrained(model_name).to(device).eval()
    processor = CLIPProcessor.from_pretrained(model_name)
    return model, processor, device


def gather_image_paths(images_dir: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths = [p for p in images_dir.rglob("*") if p.suffix.lower() in exts]
    paths.sort()
    if not paths:
        raise FileNotFoundError(f"{images_dir} içinde desteklenen uzantılarda görsel bulunamadı.")
    return paths


@torch.inference_mode()
def build_image_embeddings(
    images_dir: Path,
    out_dir: Path,
    model: CLIPModel,
    processor: CLIPProcessor,
    device: str,
    batch_size: int = 32,
    use_relative_ids: bool = True,
) -> Tuple[np.ndarray, list[str]]:
    out_dir.mkdir(parents=True, exist_ok=True)

    img_paths = gather_image_paths(images_dir)
    all_embeds: list[torch.Tensor] = []
    all_ids: list[str] = []

    for i in tqdm(range(0, len(img_paths), batch_size), desc="Embedding (image)"):
        batch_paths = img_paths[i : i + batch_size]
        images = []
        kept_paths = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                images.append(img)
                kept_paths.append(p)
            except Exception as e:
                print(f"[WARN] Skipping unreadable image: {p} ({e})")
        if not images:
            continue
        inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
        feats = model.get_image_features(**inputs)
        feats = torch.nn.functional.normalize(feats, p=2, dim=1)
        all_embeds.append(feats.cpu())
        if use_relative_ids:
            all_ids.extend([str(p.relative_to(images_dir)) for p in kept_paths])
        else:
            all_ids.extend([p.name for p in kept_paths])

    if not all_embeds:
        raise RuntimeError("Hiçbir görsel işlenemedi; görüntü yollarını kontrol edin.")

    emb = torch.cat(all_embeds, dim=0).numpy()
    np.save(out_dir / "embeddings.npy", emb)
    with open(out_dir / "ids.json", "w", encoding="utf-8") as f:
        json.dump(all_ids, f, ensure_ascii=False, indent=2)

    print(f"[INFO] embeddings shape: {emb.shape} | ids: {len(all_ids)}")
    return emb, all_ids


def parse_args():
    ap = argparse.ArgumentParser(description="CLIP ile görsel embedding üretimi")
    ap.add_argument("images_dir", type=str, help="Görsellerin bulunduğu klasör (alt klasörler dahil)")
    ap.add_argument("out_dir", type=str, help="Çıktı klasörü (embeddings.npy, ids.json)")
    ap.add_argument("--model", type=str, default="openai/clip-vit-base-patch32")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--device", type=str, default=None, choices=[None, "cpu", "cuda", "mps"])
    ap.add_argument("--abs_ids", action="store_true", help="ID olarak göreli yol yerine sadece dosya adını yaz")
    return ap.parse_args()


def main():
    args = parse_args()
    images_dir = Path(args.images_dir)
    out_dir = Path(args.out_dir)

    model, processor, device = load_clip(args.model, args.device)
    build_image_embeddings(
        images_dir,
        out_dir,
        model,
        processor,
        device,
        batch_size=args.batch_size,
        use_relative_ids=not args.abs_ids,
    )


if __name__ == "__main__":
    main()
