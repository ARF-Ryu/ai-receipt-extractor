import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest
from dotenv import load_dotenv

# Pastikan root proyek dapat diimpor saat pytest dijalankan dari folder mana pun
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

load_dotenv(ROOT_DIR / "secrets.env")
load_dotenv(ROOT_DIR / ".env")

try:
    import config
except ImportError:
    config = None

from image_processor import preprocess_receipt
from parser import AIReceiptParser
from schemas import ReceiptData

DATASET_DIR = Path(__file__).resolve().parent / "dataset"
SUPPORTED_PROVIDERS = ["openrouter", "huggingface", "local", "hybrid"]
PARSER_CACHE = {}


def _has_openrouter_key() -> bool:
    key = os.getenv("OPENROUTER_API_KEY") or (
        getattr(config, "OPENROUTER_API_KEY", "") if config else ""
    )
    return bool(str(key).strip())


def _is_local_server_reachable(base_url: str, timeout: float = 2.0) -> bool:
    models_url = f"{base_url.rstrip('/')}/models"
    try:
        with urllib.request.urlopen(models_url, timeout=timeout) as resp:
            return resp.status == 200
    except (urllib.error.URLError, TimeoutError, ValueError):
        return False


def get_dataset_samples():
    """Mengumpulkan seluruh pasangan gambar struk dan ground_truth.json di tests/dataset."""
    samples = []
    if not DATASET_DIR.exists():
        return samples

    for sample_dir in sorted(DATASET_DIR.iterdir()):
        if not sample_dir.is_dir():
            continue
        gt_path = sample_dir / "ground_truth.json"
        if not gt_path.exists():
            continue

        img_path = None
        for ext in ("receipt.jpg", "receipt.jpeg", "receipt.png"):
            candidate = sample_dir / ext
            if candidate.exists():
                img_path = candidate
                break

        if img_path is not None:
            samples.append((sample_dir.name, str(img_path), str(gt_path)))
    return samples


def pytest_generate_tests(metafunc):
    """Membuat kombinasi pengujian (sample x provider) secara dinamis sesuai flag --provider."""
    if {"sample_name", "image_path", "gt_path", "provider"}.issubset(metafunc.fixturenames):
        selected = metafunc.config.getoption("--provider").lower()
        providers = SUPPORTED_PROVIDERS if selected == "all" else [selected]

        invalid = [p for p in providers if p not in SUPPORTED_PROVIDERS]
        if invalid:
            raise ValueError(f"Provider tidak valid: {invalid}. Pilihan: {SUPPORTED_PROVIDERS + ['all']}")

        samples = get_dataset_samples()
        if not samples:
            pytest.skip(f"Tidak ada sampel uji ditemukan di {DATASET_DIR}")

        combinations = [
            (sample_name, img_path, gt_path, prov)
            for sample_name, img_path, gt_path in samples
            for prov in providers
        ]
        metafunc.parametrize("sample_name, image_path, gt_path, provider", combinations)


def _get_or_create_parser(provider: str, pytestconfig) -> AIReceiptParser:
    """Menyimpan instance model di memori selama sesi pytest agar model HF tidak dimuat berulang."""
    openrouter_model = pytestconfig.getoption("--openrouter-model")
    hf_model = pytestconfig.getoption("--hf-model")
    local_url = pytestconfig.getoption("--local-url")
    local_model = pytestconfig.getoption("--local-model")
    hybrid_backend = pytestconfig.getoption("--hybrid-backend")
    hybrid_llm_model = pytestconfig.getoption("--hybrid-llm-model")

    if provider == "openrouter":
        if not _has_openrouter_key():
            pytest.skip("OPENROUTER_API_KEY tidak tersedia; pengujian OpenRouter dilewati.")
        cache_key = ("openrouter", openrouter_model)
        if cache_key not in PARSER_CACHE:
            PARSER_CACHE[cache_key] = AIReceiptParser(
                provider="openrouter", model_name=openrouter_model
            )
        return PARSER_CACHE[cache_key]

    if provider == "local":
        if not _is_local_server_reachable(local_url):
            pytest.skip(f"Server lokal tidak aktif di {local_url}; pengujian Local dilewati.")
        cache_key = ("local", local_url, local_model)
        if cache_key not in PARSER_CACHE:
            PARSER_CACHE[cache_key] = AIReceiptParser(
                provider="local", base_url=local_url, model_name=local_model
            )
        return PARSER_CACHE[cache_key]

    if provider == "huggingface":
        cache_key = ("huggingface", hf_model)
        if cache_key not in PARSER_CACHE:
            PARSER_CACHE[cache_key] = AIReceiptParser(
                provider="huggingface", model_name=hf_model
            )
        return PARSER_CACHE[cache_key]

    if provider == "hybrid":
        if hybrid_backend == "openrouter" and not _has_openrouter_key():
            pytest.skip("OPENROUTER_API_KEY tidak tersedia untuk Stage 2 Hybrid.")
        if hybrid_backend == "local" and not _is_local_server_reachable(local_url):
            pytest.skip(f"Server lokal tidak aktif di {local_url} untuk Stage 2 Hybrid.")

        cache_key = ("hybrid", hf_model, hybrid_backend, local_url, hybrid_llm_model)
        if cache_key not in PARSER_CACHE:
            PARSER_CACHE[cache_key] = AIReceiptParser(
                provider="hybrid",
                model_name=hf_model,
                base_url=local_url,
                llm_backend=hybrid_backend,
                llm_model_name=hybrid_llm_model,
            )
        return PARSER_CACHE[cache_key]

    raise ValueError(f"Provider tidak dikenal: {provider}")


def test_receipt_extraction_across_providers(
    sample_name, image_path, gt_path, provider, pytestconfig, tmp_path
):
    with open(gt_path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    use_opencv = pytestconfig.getoption("--opencv")
    target_image = image_path
    if use_opencv:
        processed_file = tmp_path / f"{sample_name}_processed.jpg"
        target_image = preprocess_receipt(image_path, str(processed_file))

    parser = _get_or_create_parser(provider, pytestconfig)

    start_time = time.perf_counter()
    try:
        result: ReceiptData = parser.parse(target_image)
    except Exception as e:
        pytest.fail(f"[{provider} | {sample_name}] Gagal mengeksekusi parser: {e}")
    elapsed = time.perf_counter() - start_time

    # Cetak ringkasan metrik jika pytest dijalankan dengan flag -s
    print(
        f"\n[{provider.upper()} | {sample_name} | OpenCV={use_opencv}] "
        f"Waktu: {elapsed:.2f}s | Merchant: '{result.merchant_name}' | "
        f"Total: {result.total_amount} (GT: {gt_data['total_amount']}) | "
        f"Items: {len(result.line_items)}/{len(gt_data['line_items'])}"
    )

    # 1. Validasi Merchant Name
    # Catatan: Dataset CORD-v2 pada model Donut tidak memiliki label nama merchant
    hf_model_name = pytestconfig.getoption("--hf-model").lower()
    is_donut_standalone = provider == "huggingface" and "donut" in hf_model_name
    if not is_donut_standalone and gt_data.get("merchant_name"):
        assert result.merchant_name, f"[{provider} | {sample_name}] merchant_name kosong"
        gt_merchant = gt_data["merchant_name"].strip().lower()
        pred_merchant = result.merchant_name.strip().lower()
        assert gt_merchant in pred_merchant or pred_merchant in gt_merchant, (
            f"[{provider} | {sample_name}] Merchant tidak cocok: "
            f"Prediksi '{result.merchant_name}' vs GT '{gt_data['merchant_name']}'"
        )

    # 2. Validasi Total Pembayaran (toleransi 1% untuk pembulatan/pajak kecil)
    assert result.total_amount is not None, f"[{provider} | {sample_name}] total_amount bernilai None"
    assert pytest.approx(result.total_amount, rel=1e-2) == gt_data["total_amount"], (
        f"[{provider} | {sample_name}] Total salah: "
        f"Prediksi {result.total_amount} vs GT {gt_data['total_amount']}"
    )

    # 3. Validasi Jumlah Line Items
    assert len(result.line_items) == len(gt_data["line_items"]), (
        f"[{provider} | {sample_name}] Jumlah item tidak cocok: "
        f"Prediksi {len(result.line_items)} vs GT {len(gt_data['line_items'])}"
    )

    # 4. Validasi Nama dan Total Harga per Item
    for idx, (pred_item, gt_item) in enumerate(zip(result.line_items, gt_data["line_items"])):
        gt_name = gt_item["item_name"].strip().lower()
        pred_name = pred_item.item_name.strip().lower()
        assert gt_name in pred_name or pred_name in gt_name, (
            f"[{provider} | {sample_name}] Nama item #{idx} tidak cocok: "
            f"'{pred_item.item_name}' vs '{gt_item['item_name']}'"
        )
        assert pytest.approx(pred_item.item_total, rel=1e-2) == gt_item["item_total"], (
            f"[{provider} | {sample_name}] Harga item #{idx} salah: "
            f"{pred_item.item_total} vs {gt_item['item_total']}"
        )