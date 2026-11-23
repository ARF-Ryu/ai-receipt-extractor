import json
import os
import time
from typing import Any, Dict, List

from image_processor import preprocess_receipt
from parser import AIReceiptParser


def evaluate_single_prediction(pred: dict, gt: dict) -> Dict[str, Any]:
    """Menghitung metrik akurasi per field antara hasil prediksi dan ground truth."""
    pred_merchant = str(pred.get("merchant_name", "")).strip().lower()
    gt_merchant = str(gt.get("merchant_name", "")).strip().lower()
    merchant_match = (gt_merchant in pred_merchant) or (pred_merchant in gt_merchant) if gt_merchant else False

    pred_total = float(pred.get("total_amount", 0.0))
    gt_total = float(gt.get("total_amount", 0.0))
    total_diff = abs(pred_total - gt_total)
    total_match = total_diff <= 1.0  # Toleransi pembulatan 1 rupiah

    pred_items = pred.get("line_items", [])
    gt_items = gt.get("line_items", [])
    item_count_match = len(pred_items) == len(gt_items)

    return {
        "merchant_match": merchant_match,
        "total_match": total_match,
        "total_diff": total_diff,
        "pred_items_count": len(pred_items),
        "gt_items_count": len(gt_items),
        "item_count_match": item_count_match,
    }


def run_benchmark(image_path: str, ground_truth_path: str, configs: List[dict]):
    with open(ground_truth_path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    results_summary = []

    for cfg in configs:
        label = cfg["label"]
        use_opencv = cfg.get("use_opencv", True)
        target_img = preprocess_receipt(image_path, "temp_bench_proc.jpg") if use_opencv else image_path

        print(f"Menguji: {label} (OpenCV={use_opencv})...")
        start_time = time.perf_counter()
        try:
            parser = AIReceiptParser(
                provider=cfg["provider"],
                model_name=cfg.get("model_name"),
                base_url=cfg.get("base_url"),
                llm_backend=cfg.get("llm_backend", "openrouter"),
                llm_model_name=cfg.get("llm_model_name"),
            )
            result = parser.parse(target_img)
            elapsed = round(time.perf_counter() - start_time, 2)

            metrics = evaluate_single_prediction(result.model_dump(), gt_data)
            results_summary.append(
                {
                    "Konfigurasi": label,
                    "OpenCV": "Ya" if use_opencv else "Tidak",
                    "Waktu (s)": elapsed,
                    "Merchant": "PASS" if metrics["merchant_match"] else "FAIL",
                    "Total": "PASS" if metrics["total_match"] else f"FAIL (selisih {metrics['total_diff']:,.0f})",
                    "Jumlah Item": f"{metrics['pred_items_count']}/{metrics['gt_items_count']}",
                    "Raw JSON": result.model_dump(),
                }
            )
        except Exception as e:
            elapsed = round(time.perf_counter() - start_time, 2)
            results_summary.append(
                {
                    "Konfigurasi": label,
                    "OpenCV": "Ya" if use_opencv else "Tidak",
                    "Waktu (s)": elapsed,
                    "Merchant": "ERROR",
                    "Total": str(e)[:40],
                    "Jumlah Item": "-",
                    "Raw JSON": None,
                }
            )

    # Cetak tabel ringkasan Markdown siap salin
    print("\n| Konfigurasi | OpenCV | Waktu (s) | Merchant | Total | Jumlah Item |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: |")
    for r in results_summary:
        print(
            f"| {r['Konfigurasi']} | {r['OpenCV']} | {r['Waktu (s)']} | "
            f"{r['Merchant']} | {r['Total']} | {r['Jumlah Item']} |"
        )

    with open("benchmark_report1.json", "w", encoding="utf-8") as out:
        json.dump(results_summary, out, indent=2, ensure_ascii=False)
    print("\nDetail lengkap tersimpan di benchmark_report.json")


if __name__ == "__main__":
    # Sesuaikan daftar konfigurasi yang ingin Anda evaluasi
    test_configs = [
        {"label": "OpenRouter (gpt-4o-mini)", "provider": "openrouter", "model_name": "openai/gpt-4o-mini", "use_opencv": True},
        {"label": "HF SmolVLM-500M", "provider": "huggingface", "model_name": "HuggingFaceTB/SmolVLM-500M-Instruct", "use_opencv": True},
        {"label": "HF Donut CORD-v2", "provider": "huggingface", "model_name": "naver-clova-ix/donut-base-finetuned-cord-v2", "use_opencv": True},
        {"label": "Local Ollama (qwen2.5vl:3b)", "provider": "local", "base_url": "http://localhost:11434/v1", "model_name": "qwen2.5vl:3b", "use_opencv": True},
    ]

    sample_img = os.path.join("test", "datatrial", "sample1", "receipt.jpg")
    sample_gt = os.path.join("test", "datatrial", "sample1", "ground_truth.json")

    if os.path.exists(sample_img) and os.path.exists(sample_gt):
        run_benchmark(sample_img, sample_gt, test_configs)
    else:
        print("Pastikan sample_01/receipt.jpg dan ground_truth.json sudah tersedia.")