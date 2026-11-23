# 🧾 AI Receipt Extractor & Parser

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B.svg)](https://streamlit.io/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-FFD21E.svg)](https://huggingface.co/)
[![Pydantic v2](https://img.shields.io/badge/Pydantic-v2-E92063.svg)](https://docs.pydantic.dev/)
[![CI/CD](https://img.shields.io/badge/GitHub%20Actions-PyTest-2088FF.svg)](.github/workflows/ci.yml)

A modular, multi-backend receipt OCR and financial data extraction system combining **Computer Vision (OpenCV)**, **Cloud Vision LLMs (OpenRouter)**, **Local Inference Servers (Ollama / LM Studio / vLLM)**, and **Hugging Face Vision-Language Models (SmolVLM & Donut)**. Built to transform noisy thermal receipts into validated, type-safe JSON schemas.

---

## 🏗️ Supported Inference Architectures

| Mode | Engine / Models | Best Use Case |
| :--- | :--- | :--- |
| **1. Cloud Vision Gateway** | **OpenRouter** (`openai/gpt-4o-mini`, Qwen-VL, Gemini, Claude) | Lowest latency (~9–10s) and strong out-of-the-box schema compliance across diverse layouts. |
| **2. Local Vision Server** | **Ollama** (`:11434/v1`), **LM Studio** (`:1234/v1`), **vLLM** (`:8000/v1`) | Privacy-first local execution using quantized models (e.g., `qwen2.5vl:3b`, `granite3.2-vision:2b`). |
| **3. Standalone Hugging Face** | `HuggingFaceTB/SmolVLM-500M-Instruct`, `naver-clova-ix/donut-base-finetuned-cord-v2` | Self-contained local PyTorch inference without external server dependencies; fits within 4GB VRAM GPUs. |
| **4. Hybrid Two-Stage Pipeline** | **Stage 1:** HF Local VLM/OCR → **Stage 2:** Text LLM (OpenRouter / Local) | Separates visual text transcription from JSON structuring to reduce vision token usage and enforce schema adherence. |

---

## 📁 Project Structure

```text
custom-receipt-ai/
│
├── .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions CI/CD workflow for automated PyTest runs
├── schemas.py                  # Pydantic v2 data contracts (ReceiptData & LineItem)
├── image_processor.py          # OpenCV grayscale, denoising & adaptive thresholding pipeline
├── parser.py                   # Unified multi-provider AI engine & JSON normalizer
├── app.py                      # Interactive Streamlit web dashboard with model caching
├── benchmark.py                # Multi-model evaluation & latency benchmarking script
├── config.example.py           # Configuration template for local API keys
├── .gitignore                  # Excludes secrets, virtual environments, and temp files
├── requirements.txt            # Python dependencies
└── test/                      # Multi-provider AI Evaluation & Regression Suite
    ├── conftest.py             # Custom PyTest CLI flags (--provider, --hf-model, --opencv, etc.)
    ├── test_receipt_parser.py  # Automated field-level accuracy & numeric tolerance tests
    └── datatrial/
        └── sample1/
            ├── receipt.jpg
            └── ground_truth.json
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/ARF-Ryu/custom-receipt-ai.git](https://github.com/ARF-Ryu/custom-receipt-ai.git)
cd custom-receipt-ai
```

### 2. Create and Activate Virtual Environment
```bash
python -m venv venv

# Windows:
venv\Scripts\activate

# Linux / macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
> **GPU Acceleration Note:** If you have an NVIDIA GPU with CUDA, install the CUDA-enabled build of PyTorch from [pytorch.org](https://pytorch.org/) before running local Hugging Face inference.

### 4. Configure API Keys
Create a `secrets.env` file in the project root (or copy `config.example.py` to `config.py`):
```env
OPENROUTER_API_KEY="sk-or-v1-your-api-key-here"
```
> **Local Server Note:** No API key is required when running **Ollama** (`http://localhost:11434/v1`) or **LM Studio** (`http://localhost:1234/v1`). Ensure the local server is active and your vision model is pulled/loaded before running extraction.

---

## 🚀 Usage

### 1. Interactive Web Dashboard (Streamlit)
Launch the web UI to switch between providers, toggle OpenCV preprocessing, inspect raw model outputs, and export structured JSON:
```bash
streamlit run app.py
```

### 2. Automated Multi-Provider Evaluation (PyTest)
Validate extraction accuracy (merchant name, total amount within 1% tolerance, and itemized line items) against `tests/dataset/`:

* **Test OpenRouter (Default):**
  ```bash
  pytest -v -s --provider=openrouter
  ```
* **Test Hugging Face Models (SmolVLM or Donut CORD-v2):**
  ```bash
  pytest -v -s --provider=huggingface --hf-model="HuggingFaceTB/SmolVLM-500M-Instruct"
  pytest -v -s --provider=huggingface --hf-model="naver-clova-ix/donut-base-finetuned-cord-v2"
  ```
* **Test Local Server (Ollama / LM Studio):**
  ```bash
  pytest -v -s --provider=local --local-url="http://localhost:11434/v1" --local-model="qwen2.5vl:3b"
  ```
* **Test Hybrid Two-Stage Pipeline:**
  ```bash
  pytest -v -s --provider=hybrid --hf-model="HuggingFaceTB/SmolVLM-500M-Instruct" --hybrid-backend="openrouter"
  ```
* **Run Full Benchmark Across All Available Providers (with OpenCV Preprocessing):**
  ```bash
  pytest -v -s --provider=all --opencv
  ```

### 3. Quantitative Multi-Model Benchmark
Run `benchmark.py` to generate a Markdown comparison table and export detailed JSON telemetry (`benchmark_report.json`):
```bash
python benchmark.py
```

---

## 📊 Benchmark & Evaluation Results

All four inference backends were evaluated on an Indonesian retail thermal receipt (`TLOGOMAS 44 MALANG`, 9 ground-truth items, final total `Rp 75,400` after a `Rp 2,000` discount) on a **4GB VRAM GPU** setup, comparing raw RGB inputs (`OpenCV: Off`) against binarized inputs (`OpenCV: On`).

### 1. Quantitative Comparison

| Configuration | OpenCV | Latency (s) | Merchant | Total Amount | Items Detected | Field-Level Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **OpenRouter (`gpt-4o-mini`)** | Off | **8.77s** | PASS | PASS | 8/9 | Fastest execution; merged last two similarly named items (`KNZLER`/`KANZLR`). |
| **OpenRouter (`gpt-4o-mini`)** | On | 10.15s | PASS | PASS | **9/9** | Recovered all 9 rows & full header, but binarization eroded 2 numeric values (`4900` → `490`, qty `2` → `1`). |
| **Local Ollama (`qwen2.5vl:3b`)** | Off | 147.53s | PASS | PASS | **9/9** | **100% precision** across header, dates, unit prices, quantities, and final total. |
| **Local Ollama (`qwen2.5vl:3b`)** | On | 149.45s | PASS | PASS | **9/9** | **100% precision** maintained; resilient against binarization artifacts. |
| **HF `Donut CORD-v2`** | Off | 27.96s | ERROR | ERROR | - | Crashed converting postal code string (`65144...`) to `float` in item quantity (fixed in `_to_float`). |
| **HF `Donut CORD-v2`** | On | 33.00s | FAIL | FAIL | 9/9 | Extracted 8/9 product items accurately, but absorbed store header/phone into item #1. |
| **HF `SmolVLM-500M`** | Off | 191.27s | PASS | FAIL | 15/9 | Experienced schema drift (outputting CORD keys `nm`/`cnt`) and truncated mid-JSON. |
| **HF `SmolVLM-500M`** | On | 165.61s | PASS | FAIL | 15/9 | Faster decoding, but still drifted from target Pydantic keys into fallback line splitting. |

### 2. Model Strengths, Limitations & Selection Guide

| Model / Backend | Strengths | Limitations | Recommended Use Case |
| :--- | :--- | :--- | :--- |
| **`qwen2.5vl:3b` (Ollama Local)** | Highest overall accuracy (**100%** on all line items and math); full data privacy; immune to OpenCV thresholding noise. | High latency (~148s on 4GB VRAM due to KV-cache memory pressure and cold model loading). | Offline/private financial processing where **accuracy and privacy** matter more than real-time speed. |
| **`gpt-4o-mini` (OpenRouter Cloud)** | Very fast (**~8.8–10.1s**); zero local GPU load; accurate date/time/address extraction. | Can merge adjacent rows with similar names or misread thin digits if aggressive thresholding is applied. | Production web apps requiring **low latency** and high throughput (best used **without** binary thresholding). |
| **`donut-base-finetuned-cord-v2` (HF)** | Fast local execution (~28–33s) without an external server; strong item/price extraction on clean layouts. | Not trained on merchant names/addresses (causes header-to-item shift); misreads totals on Indonesian receipts. | Lightweight local itemization or as a base checkpoint for **custom fine-tuning** on local receipts. |
| **`SmolVLM-500M-Instruct` (HF)** | Ultra-compact footprint (~1GB); reads raw product text well visually. | Sub-1B capacity struggles with strict multi-nested JSON adherence (`schema drift` & token repetition). | Best used only as **Stage 1 OCR** inside the **Hybrid Pipeline** paired with a stronger Text LLM. |

### 3. Key Takeaways on OpenCV Preprocessing
* **Natural RGB is Optimal for Modern VLMs:** Vision-Language Transformers (`gpt-4o-mini`, `qwen2.5vl:3b`) are trained on natural RGB encoders. Applying aggressive `adaptiveThreshold` binarization can erode thin thermal digits (e.g., turning `4900` into `490` or `LE MINERALE` qty `2` into `1`).
* **When to Enable OpenCV:** Keep OpenCV preprocessing enabled primarily for **Donut CORD-v2** or severely faded/low-contrast receipts where background thermal noise prevents line segmentation.

---

## 🗺️ Future Engineering Roadmap

1. **Cross-Field Mathematical Validation (`Pydantic @model_validator`):**
   Add an automatic reconciliation check comparing $\sum (\text{item\_quantity} \times \text{item\_price})$ against `total_amount`. Flag discrepancies exceeding expected tax/discount thresholds with `needs_review=True` or trigger an automatic retry.
2. **Item-Level Granular Metrics in `benchmark.py`:**
   Expand evaluation assertions beyond `total_amount` and `len(line_items)` to compute **per-item price/quantity F1-score** and **Character Error Rate (CER)** on merchant and product names.
3. **4GB VRAM Latency Optimization for Local Ollama:**
   Downscale input images to a maximum edge of **1024–1280px** before encoding and set `num_ctx: 2048` in Ollama requests to prevent VRAM spillover into system RAM and cut `qwen2.5vl:3b` latency from ~148s to under 25s.
4. **Header-Filtering Post-Processor for Donut:**
   Add a heuristic filter in `_map_donut_cord_to_receipt()` to discard top-of-receipt store names/phone numbers mistakenly classified as `menu[0]`.
5. **QLoRA Fine-Tuning on Indonesian Receipts:**
   Fine-tune `Qwen2.5-VL-3B` or `Donut` on a curated dataset of Indonesian retail receipts (Alfamart, Indomaret, local POS) to eliminate header drift and discount miscalculations.

---

## 📝 License

This project is licensed under the [MIT License](LICENSE).
