"""
Farklı CLIP modellerini karşılaştır
"""
import subprocess
import json
from pathlib import Path

models = [
    "openai/clip-vit-base-patch32",      # Mevcut (base)
    "openai/clip-vit-large-patch14",     # Daha büyük
    "openai/clip-vit-base-patch16",      # Farklı patch size
]

results = []

for model in models:
    print(f"\n{'='*60}")
    print(f"🤖 Model: {model}")
    print('='*60)
    
    # Model ID'yi environment variable olarak set et
    # Windows PowerShell için
    cmd = f'$env:CLIP_MODEL="{model}"; python scripts/evaluate.py --num_queries 50 --k_values 1 5 10'
    
    try:
        result = subprocess.run(
            ["powershell", "-Command", cmd],
            capture_output=True,
            text=True,
            cwd="."
        )
        
        if result.returncode == 0:
            print(result.stdout)
            
            # JSON sonuçları oku
            eval_file = Path("artifacts/evaluation_results.json")
            if eval_file.exists():
                with open(eval_file, 'r') as f:
                    data = json.load(f)
                    results.append({
                        'model': model,
                        'recall': data['summary']['recall'],
                        'latency_ms': data['summary']['latency_mean_ms']
                    })
        else:
            print(f"❌ Error: {result.stderr}")
    except Exception as e:
        print(f"❌ Exception: {e}")

# Sonuçları karşılaştır
print("\n" + "="*70)
print("📊 MODEL KARŞILAŞTIRMASI")
print("="*70)

for res in results:
    print(f"\n🤖 {res['model']}")
    print(f"   Recall@1:  {res['recall'].get(1, 0)*100:.2f}%")
    print(f"   Recall@5:  {res['recall'].get(5, 0)*100:.2f}%")
    print(f"   Recall@10: {res['recall'].get(10, 0)*100:.2f}%")
    print(f"   Latency:   {res['latency_ms']:.2f} ms")

# En iyiyi bul
best = max(results, key=lambda x: x['recall'].get(5, 0))
print(f"\n🏆 En İyi Model: {best['model']}")
print(f"   Recall@5: {best['recall'].get(5, 0)*100:.2f}%")
