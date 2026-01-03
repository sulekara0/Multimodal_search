"""
Multimodal Search Evaluation Script

Metriks:
- Recall@K: İlk K sonuç içinde doğru görsellerin oranı
- nDCG@K: Normalized Discounted Cumulative Gain
- Latency: Ortalama ve p95 sorgu süresi
"""

import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor
import faiss
from collections import defaultdict
from tqdm import tqdm

# Paths
ARTIFACTS = Path("artifacts")
IMG_EMBEDDINGS = ARTIFACTS / "img_embeddings.npy"
IMG_IDS = ARTIFACTS / "img_ids.json"
FAISS_INDEX = ARTIFACTS / "faiss_index.bin"
FAISS_META = ARTIFACTS / "faiss_meta.json"
COCO_CAPTIONS_VAL = Path("D:/COCO_Dataset_2017/archive/coco2017/annotations/captions_val2017.json")
COCO_CAPTIONS_TRAIN = Path("D:/COCO_Dataset_2017/archive/coco2017/annotations/captions_train2017.json")


def load_coco_captions(captions_paths: List[Path]) -> Dict[str, List[str]]:
    """COCO captions.json'dan image_id -> captions mapping oluştur (birden fazla dosyadan)"""
    filename_to_captions = {}
    
    for captions_path in captions_paths:
        if not captions_path.exists():
            print(f"   ⚠️  Atlanıyor (bulunamadı): {captions_path}")
            continue
            
        print(f"   📖 Yükleniyor: {captions_path.name}")
        with open(captions_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # image_id -> captions
        img_to_captions = defaultdict(list)
        for ann in data['annotations']:
            img_id = ann['image_id']
            caption = ann['caption'].strip()
            img_to_captions[img_id].append(caption)
        
        # image_id -> filename mapping
        img_id_to_filename = {}
        for img in data['images']:
            img_id_to_filename[img['id']] = img['file_name']
        
        # filename -> captions (birleştir)
        for img_id, captions in img_to_captions.items():
            filename = img_id_to_filename.get(img_id)
            if filename:
                if filename in filename_to_captions:
                    # Zaten varsa, caption'ları ekle
                    filename_to_captions[filename].extend(captions)
                else:
                    filename_to_captions[filename] = captions
        
        print(f"      ✅ {len(img_to_captions)} görsel caption'ı eklendi")
    
    return filename_to_captions


def load_faiss_system():
    """FAISS index ve metadata yükle"""
    index = faiss.read_index(str(FAISS_INDEX))
    
    with open(FAISS_META, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    metric = meta.get('metric', 'ip')
    
    with open(IMG_IDS, 'r', encoding='utf-8') as f:
        img_ids = json.load(f)
    
    return index, metric, img_ids


def load_clip_model(model_name: str = "openai/clip-vit-base-patch32"):
    """CLIP model yükle"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(model_name).to(device).eval()
    processor = CLIPProcessor.from_pretrained(model_name)
    return model, processor, device


def encode_text(text: str, model, processor, device):
    """Text'i CLIP ile encode et"""
    with torch.inference_mode():
        inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        features = model.get_text_features(**inputs).detach().cpu().numpy().astype("float32")
    return features


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """L2 normalize"""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
    return embeddings / norms


def search(query: str, index, metric: str, img_ids: List[str], 
           model, processor, device, k: int = 10) -> Tuple[List[str], List[float], float]:
    """Text query ile arama yap ve latency ölç"""
    t0 = time.perf_counter()
    
    # Encode query
    query_emb = encode_text(query, model, processor, device)
    if metric == "ip":
        query_emb = normalize_embeddings(query_emb)
        scores, indices = index.search(query_emb, k)
    else:  # l2
        scores, indices = index.search(query_emb, k)
        scores = -scores  # negatif = daha iyi
    
    latency = time.perf_counter() - t0
    
    results = [img_ids[i] for i in indices[0]]
    scores_list = scores[0].tolist()
    
    return results, scores_list, latency


def calculate_recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """Recall@K hesapla"""
    if not relevant:
        return 0.0
    retrieved_at_k = set(retrieved[:k])
    relevant_set = set(relevant)
    hits = len(retrieved_at_k & relevant_set)
    return hits / len(relevant_set)


def calculate_ndcg_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """nDCG@K hesapla"""
    if not relevant:
        return 0.0
    
    # DCG
    dcg = 0.0
    for i, item in enumerate(retrieved[:k]):
        if item in relevant:
            dcg += 1.0 / np.log2(i + 2)  # i+2 çünkü 0-indexed
    
    # IDCG (ideal DCG)
    idcg = 0.0
    for i in range(min(len(relevant), k)):
        idcg += 1.0 / np.log2(i + 2)
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def create_test_queries(filename_to_captions: Dict[str, List[str]], 
                        img_ids: List[str], 
                        num_queries: int = 100) -> List[Tuple[str, List[str]]]:
    """Test query'leri oluştur (caption -> ilgili görseller)"""
    queries = []
    
    # img_ids'den filename'leri normalize et (path varsa basename al)
    normalized_ids = [Path(img_id).name for img_id in img_ids]
    
    available_files = set(normalized_ids)
    
    count = 0
    for filename, captions in filename_to_captions.items():
        if filename not in available_files:
            continue
        
        # İlk caption'ı query olarak kullan
        query = captions[0]
        # Bu görselin filename'i relevant
        relevant = [filename]
        
        queries.append((query, relevant))
        count += 1
        
        if count >= num_queries:
            break
    
    return queries


def evaluate_system(queries: List[Tuple[str, List[str]]], 
                   index, metric: str, img_ids: List[str],
                   model, processor, device,
                   k_values: List[int] = [1, 5, 10]):
    """Sistemin tam değerlendirmesi"""
    
    results = {
        'recall': {k: [] for k in k_values},
        'ndcg': {k: [] for k in k_values},
        'latencies': []
    }
    
    print(f"\n🔍 {len(queries)} query ile değerlendirme başlıyor...")
    
    for query, relevant in tqdm(queries, desc="Evaluating"):
        # Normalize relevant filenames
        relevant_normalized = [Path(r).name for r in relevant]
        
        # Search
        retrieved, scores, latency = search(
            query, index, metric, img_ids, model, processor, device, k=max(k_values)
        )
        
        # Normalize retrieved filenames
        retrieved_normalized = [Path(r).name for r in retrieved]
        
        results['latencies'].append(latency)
        
        # Her K değeri için metrikleri hesapla
        for k in k_values:
            recall = calculate_recall_at_k(retrieved_normalized, relevant_normalized, k)
            ndcg = calculate_ndcg_at_k(retrieved_normalized, relevant_normalized, k)
            results['recall'][k].append(recall)
            results['ndcg'][k].append(ndcg)
    
    # Özet istatistikler
    summary = {
        'recall': {k: np.mean(results['recall'][k]) for k in k_values},
        'ndcg': {k: np.mean(results['ndcg'][k]) for k in k_values},
        'latency_mean_ms': np.mean(results['latencies']) * 1000,
        'latency_p95_ms': np.percentile(results['latencies'], 95) * 1000,
        'latency_p99_ms': np.percentile(results['latencies'], 99) * 1000,
        'num_queries': len(queries)
    }
    
    return summary, results


def print_results(summary: Dict):
    """Sonuçları formatla ve yazdır"""
    print("\n" + "="*60)
    print("📊 EVALUATION RESULTS")
    print("="*60)
    
    print(f"\n📈 Recall@K:")
    for k, value in summary['recall'].items():
        print(f"   Recall@{k:2d}: {value:.4f} ({value*100:.2f}%)")
    
    print(f"\n📈 nDCG@K:")
    for k, value in summary['ndcg'].items():
        print(f"   nDCG@{k:2d}:   {value:.4f}")
    
    print(f"\n⏱️  Latency:")
    print(f"   Mean:  {summary['latency_mean_ms']:.2f} ms")
    print(f"   P95:   {summary['latency_p95_ms']:.2f} ms")
    print(f"   P99:   {summary['latency_p99_ms']:.2f} ms")
    
    print(f"\n📝 Test Queries: {summary['num_queries']}")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Multimodal Search Evaluation")
    parser.add_argument("--captions_val", type=str, default=str(COCO_CAPTIONS_VAL),
                       help="Path to COCO captions_val2017.json")
    parser.add_argument("--captions_train", type=str, default=str(COCO_CAPTIONS_TRAIN),
                       help="Path to COCO captions_train2017.json")
    parser.add_argument("--num_queries", type=int, default=100,
                       help="Number of test queries")
    parser.add_argument("--k_values", type=int, nargs='+', default=[1, 5, 10],
                       help="K values for Recall@K and nDCG@K")
    parser.add_argument("--output", type=str, default="artifacts/evaluation_results.json",
                       help="Output JSON file for detailed results")
    
    args = parser.parse_args()
    
    print("🚀 Evaluation başlatılıyor...")
    
    # Load COCO captions (val + train)
    print(f"📖 COCO captions yükleniyor...")
    captions_paths = [Path(args.captions_val), Path(args.captions_train)]
    filename_to_captions = load_coco_captions(captions_paths)
    print(f"   ✅ Toplam {len(filename_to_captions)} görsel için caption yüklendi")
    
    # Load FAISS system
    print("📦 FAISS index yükleniyor...")
    index, metric, img_ids = load_faiss_system()
    print(f"   ✅ {len(img_ids)} görsel index'te")
    
    # Load CLIP
    print("🤖 CLIP model yükleniyor...")
    model, processor, device = load_clip_model()
    print(f"   ✅ Model yüklendi (device: {device})")
    
    # Create test queries
    print(f"📝 Test query'leri oluşturuluyor ({args.num_queries} adet)...")
    queries = create_test_queries(filename_to_captions, img_ids, args.num_queries)
    print(f"   ✅ {len(queries)} query hazır")
    
    # Evaluate
    summary, detailed_results = evaluate_system(
        queries, index, metric, img_ids, model, processor, device, args.k_values
    )
    
    # Print results
    print_results(summary)
    
    # Save to file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        'summary': summary,
        'config': {
            'num_queries': len(queries),
            'k_values': args.k_values,
            'metric': metric,
            'num_images': len(img_ids)
        }
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Detaylı sonuçlar kaydedildi: {output_path}")


if __name__ == "__main__":
    main()
