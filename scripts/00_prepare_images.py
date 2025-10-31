import os, shutil, hashlib
from pathlib import Path
from PIL import Image, ImageOps

# Kaynak klasör(ler): elindeki dağınık foto klasörlerini buraya ekle
SOURCE_DIRS = [
    r"C:\Users\minions\Desktop\cars",   # örnek
    r"C:\Users\minions\Desktop\beach"   # örnek
]
TARGET_ROOT = Path("data/images")       # proje içindeki hedef

# İsteğe bağlı: Bu sözlük, klasör adından kategori çıkarır
# Örn: C:\...\Desktop\cars  -> "cars"
def infer_category_from_path(p: Path) -> str:
    return p.name.lower()

# Uzantılar
VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
MAX_SIDE = 1024  # uzun kenar limiti

def file_hash(path: Path, chunk=65536) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b: break
            h.update(b)
    return h.hexdigest()

def process_image(src: Path, dst: Path):
    # Pillow ile aç, auto-orient + RGB + resize, JPEG olarak kaydet
    with Image.open(src) as im:
        im = ImageOps.exif_transpose(im)      # EXIF rotasyonunu uygula
        if im.mode not in ("RGB", "L"):
            im = im.convert("RGB")
        # Boyutlandır
        w, h = im.size
        scale = MAX_SIDE / max(w, h) if max(w, h) > MAX_SIDE else 1.0
        if scale < 1.0:
            im = im.resize((int(w*scale), int(h*scale)))
        # JPEG’e yaz
        dst.parent.mkdir(parents=True, exist_ok=True)
        im.save(dst.with_suffix(".jpg"), format="JPEG", quality=90, optimize=True)

def main():
    seen_hashes = set()
    count_in, count_out, count_dup, count_err = 0, 0, 0, 0

    for sdir in SOURCE_DIRS:
        sdir = Path(sdir)
        category = infer_category_from_path(sdir)
        for path in sdir.rglob("*"):
            if not path.is_file(): continue
            if path.suffix.lower() not in VALID_EXTS: continue
            count_in += 1
            try:
                h = file_hash(path)
                if h in seen_hashes:
                    count_dup += 1
                    continue
                seen_hashes.add(h)
                # Hedef dosya adı: kategori + orijinal ad (çakışırsa hash ekle)
                dst_dir = TARGET_ROOT / category
                dst_name = path.stem
                dst = dst_dir / (dst_name + ".jpg")
                i = 1
                while dst.exists():
                    dst = dst_dir / f"{dst_name}_{i}.jpg"
                    i += 1
                process_image(path, dst)
                count_out += 1
            except Exception as e:
                print(f"⚠️ Hata: {path} -> {e}")
                count_err += 1

    print(f"✅ Bitti | Girdi: {count_in}  Çıktı: {count_out}  Çift: {count_dup}  Hata: {count_err}")

if __name__ == "__main__":
    main()
