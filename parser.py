import base64
import json
import os
import re
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI, APIConnectionError, AuthenticationError, NotFoundError

try:
    import config
except ImportError:
    config = None

from schemas import LineItem, ReceiptData

# Memuat secrets.env jika ada (menghindari bentrok jika folder venv bernama .env)
load_dotenv("secrets.env")
load_dotenv()


def _extract_json_payload(content: str) -> dict:
    """Membersihkan teks respons model dan mengambil blok JSON utama."""
    cleaned = content.strip()
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned, re.IGNORECASE)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    start_idx = cleaned.find("{")
    end_idx = cleaned.rfind("}")
    if start_idx != -1 and end_idx != -1 and end_idx >= start_idx:
        cleaned = cleaned[start_idx : end_idx + 1]

    return json.loads(cleaned)


def _to_float(value, default: float = 0.0) -> float:
    """Mengonversi string mata uang (misal 'Rp 25.000' atau '25,000.50') menjadi float aman."""
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)

    text = re.sub(r"[^\d.,-]", "", str(value)).strip()
    if not text:
        return default

    if "." in text and "," not in text:
        parts = text.split(".")
        if len(parts[-1]) == 3:
            text = text.replace(".", "")
    else:
        text = text.replace(",", "")

    try:
        return float(text)
    except ValueError:
        return default


def _normalize_receipt_dict(raw: dict) -> dict:
    """Menormalisasi hasil JSON model agar selalu lolos validasi skema Pydantic ReceiptData."""
    normalized_items = []
    for item in raw.get("line_items") or []:
        if not isinstance(item, dict):
            continue
        qty = _to_float(item.get("item_quantity"), default=1.0)
        price = _to_float(item.get("item_price"), default=0.0)
        total = _to_float(item.get("item_total"), default=qty * price)
        normalized_items.append(
            {
                "item_name": str(item.get("item_name") or "Item").strip(),
                "item_quantity": qty,
                "item_price": price,
                "item_total": total,
            }
        )

    total_amount = _to_float(raw.get("total_amount"), default=0.0)
    if total_amount == 0.0 and normalized_items:
        total_amount = sum(i["item_total"] for i in normalized_items)

    return {
        "merchant_name": str(raw.get("merchant_name") or "Unknown Merchant").strip(),
        "merchant_address": raw.get("merchant_address"),
        "transaction_date": raw.get("transaction_date"),
        "transaction_time": raw.get("transaction_time"),
        "total_amount": total_amount,
        "line_items": normalized_items,
    }


def _map_donut_cord_to_receipt(cord_json: dict) -> ReceiptData:
    """Mengonversi output JSON bawaan model Donut CORD-v2 ke skema ReceiptData."""
    raw_menu = cord_json.get("menu", [])
    if isinstance(raw_menu, dict):
        raw_menu = [raw_menu]

    items = []
    for entry in raw_menu:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("nm") or "Item").strip()
        qty = _to_float(entry.get("cnt"), default=1.0)
        if qty <= 0:
            qty = 1.0
        total_price = _to_float(entry.get("price"), default=0.0)
        unit_price = _to_float(
            entry.get("unitprice"),
            default=(total_price / qty if qty > 0 else total_price),
        )
        items.append(
            LineItem(
                item_name=name,
                item_quantity=qty,
                item_price=unit_price,
                item_total=total_price,
            )
        )

    total_block = cord_json.get("total", {})
    if isinstance(total_block, list) and total_block:
        total_block = total_block[0]
    total_amount = _to_float(total_block.get("total_price") if isinstance(total_block, dict) else 0.0)

    if total_amount == 0.0 and items:
        total_amount = sum(item.item_total for item in items)

    return ReceiptData(
        merchant_name="Extracted via Donut (CORD-v2)",
        merchant_address=None,
        transaction_date=None,
        transaction_time=None,
        total_amount=total_amount,
        line_items=items,
    )

def _fallback_parse_unstructured_text(raw_text: str) -> ReceiptData:
    """Menyelamatkan data hasil ekstraksi jika model lokal kecil menghasilkan JSON yang terpotong."""
    merchant_match = re.search(r'"merchant_name"\s*:\s*"([^"]+)"', raw_text)
    date_match = re.search(r'"transaction_date"\s*:\s*"([^"]+)"', raw_text)
    total_match = re.search(r'"total_amount"\s*:\s*"?([\d.,]+)"?', raw_text)

    items = []
    for m in re.finditer(
        r'"item_name"\s*:\s*"([^"]+)"[^{}]*?"item_total"\s*:\s*"?([\d.,]+)"?',
        raw_text,
        re.DOTALL,
    ):
        price = _to_float(m.group(2))
        items.append(LineItem(item_name=m.group(1).strip(), item_quantity=1.0, item_price=price, item_total=price))

    if not items:
        lines = [
            ln.strip()
            for ln in raw_text.splitlines()
            if ln.strip() and not ln.strip().startswith(("{", "}", "[", "]"))
        ]
        items = [LineItem(item_name=ln[:120], item_quantity=1.0, item_price=0.0, item_total=0.0) for ln in lines[:15]]

    total_val = _to_float(total_match.group(1)) if total_match else sum(i.item_total for i in items)
    return ReceiptData(
        merchant_name=merchant_match.group(1).strip() if merchant_match else "Extracted via Local VLM",
        merchant_address=None,
        transaction_date=date_match.group(1).strip() if date_match else None,
        transaction_time=None,
        total_amount=total_val,
        line_items=items,
    )

class AIReceiptParser:
    def __init__(
        self,
        provider: str = "openrouter",
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        llm_backend: str = "openrouter",
        llm_model_name: Optional[str] = None,
    ):
        self.provider = provider.lower()
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key
        self.llm_backend = llm_backend.lower()
        self.llm_model_name = llm_model_name
        self.last_raw_response: Optional[str] = None

        if self.provider == "openrouter":
            self.client = self._build_openrouter_client()
            if not self.model_name:
                self.model_name = "openai/gpt-4o-mini"

        elif self.provider in ("local", "ollama", "lmstudio"):
            target_url = (self.base_url or "http://localhost:1234/v1").strip()
            self.client = OpenAI(base_url=target_url, api_key="local-no-key")
            if not self.model_name:
                self.model_name = "qwen2.5vl:3b"

        elif self.provider == "huggingface":
            self._init_hf_model(self.model_name or "HuggingFaceTB/SmolVLM-500M-Instruct")

        elif self.provider == "hybrid":
            # Stage 1: Model Vision lokal Hugging Face untuk membaca teks struk
            self._init_hf_model(self.model_name or "HuggingFaceTB/SmolVLM-500M-Instruct")

            # Stage 2: Text LLM (OpenRouter atau Server Lokal) untuk menyusun JSON terstruktur
            if self.llm_backend == "openrouter":
                self.client = self._build_openrouter_client()
                if not self.llm_model_name:
                    self.llm_model_name = "openai/gpt-4o-mini"
            else:
                target_url = (self.base_url or "http://localhost:1234/v1").strip()
                self.client = OpenAI(base_url=target_url, api_key="local-no-key")
                if not self.llm_model_name:
                    self.llm_model_name = "qwen2.5vl:3b"
        else:
            raise ValueError(f"Provider tidak didukung: {provider}")

    def _build_openrouter_client(self) -> OpenAI:
        raw_key = (
            self.api_key
            or os.getenv("OPENROUTER_API_KEY")
            or (getattr(config, "OPENROUTER_API_KEY", "") if config else "")
        )
        cleaned_key = str(raw_key).strip() if raw_key else ""
        if not cleaned_key:
            raise ValueError(
                "OPENROUTER_API_KEY tidak ditemukan. Isi di config.py, file secrets.env, atau kolom API Key di sidebar."
            )

        base_url = (
            getattr(config, "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
            if config
            else "https://openrouter.ai/api/v1"
        ).strip()

        return OpenAI(
            base_url=base_url,
            api_key=cleaned_key,
            default_headers={
                "HTTP-Referer": "https://github.com/custom-receipt-ai",
                "X-Title": "AI Receipt Extractor",
            },
        )

    def _init_hf_model(self, hf_model_id: str):
        import torch

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.hf_model_id = hf_model_id

        if "donut" in self.hf_model_id.lower():
            from transformers import DonutProcessor, VisionEncoderDecoderModel

            self.hf_arch = "donut"
            self.processor = DonutProcessor.from_pretrained(self.hf_model_id)
            self.model = VisionEncoderDecoderModel.from_pretrained(self.hf_model_id).to(self.device)
            if self.device == "cuda":
                self.model = self.model.half()
            self.model.eval()
        else:
            # Arsitektur native transformers v5 (SmolVLM, Qwen2.5-VL, Gemma-3, dll.)
            from transformers import AutoModelForImageTextToText, AutoProcessor

            self.hf_arch = "vlm_native"
            self.processor = AutoProcessor.from_pretrained(self.hf_model_id)
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.hf_model_id,
                torch_dtype=self.torch_dtype,
            ).to(self.device)
            self.model.eval()

    def _run_hf_vlm(
        self,
        image_path: str,
        prompt_text: str,
        max_new_tokens: int = 512,
        json_prefill: bool = False,
    ) -> str:
        """Menjalankan inferensi lokal pada model Vision-Language native Hugging Face."""
        import torch
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        chat_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        prefill_str = '{"merchant_name": "' if json_prefill else ""
        if prefill_str:
            chat_text = chat_text + prefill_str

        inputs = self.processor(text=chat_text, images=[image], return_tensors="pt")
        inputs = {
            k: v.to(self.device, dtype=self.torch_dtype) if v.is_floating_point() else v.to(self.device)
            for k, v in inputs.items()
        }

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.15,
                no_repeat_ngram_size=12,
            )

        generated_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        decoded = self.processor.batch_decode(generated_trimmed, skip_special_tokens=True)[0]
        return prefill_str + decoded

    def _run_donut(self, image_path: str) -> ReceiptData:
        """Menjalankan ekstraksi receipt-to-JSON pada model Donut CORD-v2."""
        import torch
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        task_prompt = "<s_cord-v2>"
        decoder_input_ids = self.processor.tokenizer(
            task_prompt, add_special_tokens=False, return_tensors="pt"
        ).input_ids.to(self.device)
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.to(
            self.device, dtype=self.torch_dtype
        )

        with torch.no_grad():
            outputs = self.model.generate(
                pixel_values,
                decoder_input_ids=decoder_input_ids,
                max_length=self.model.decoder.config.max_position_embeddings,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                use_cache=True,
                bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
                return_dict_in_generate=True,
            )

        sequence = self.processor.batch_decode(outputs.sequences)[0]
        sequence = sequence.replace(self.processor.tokenizer.eos_token, "").replace(
            self.processor.tokenizer.pad_token, ""
        )
        sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()
        parsed_cord = self.processor.token2json(sequence)
        self.last_raw_response = json.dumps(parsed_cord, indent=2)
        return _map_donut_cord_to_receipt(parsed_cord)

    def _structure_text_with_llm(self, ocr_text: str) -> ReceiptData:
        """Stage 2 Hybrid: Menyusun teks mentah OCR menjadi JSON terstruktur via Text LLM."""
        prompt = (
            "Susun teks mentah hasil OCR struk belanja berikut menjadi JSON murni tanpa teks pengantar "
            "dengan struktur kunci persis berikut:\n"
            '{"merchant_name": "string", "merchant_address": "string atau null", '
            '"transaction_date": "YYYY-MM-DD atau null", "transaction_time": "HH:MM:SS atau null", '
            '"total_amount": 0.0, "line_items": [{"item_name": "string", "item_quantity": 1, '
            '"item_price": 0.0, "item_total": 0.0}]}\n'
            "Pastikan semua angka berupa angka murni tanpa simbol mata uang.\n\n"
            f"=== TEKS OCR ===\n{ocr_text}"
        )

        request_kwargs = {
            "model": self.llm_model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
        }
        if self.llm_backend == "openrouter":
            request_kwargs["response_format"] = {"type": "json_object"}

        response = self._call_openai_compatible(request_kwargs, self.llm_model_name)
        raw_content = response.choices[0].message.content or "{}"
        self.last_raw_response = (
            f"--- [STAGE 1: LOCAL VLM OCR] ---\n{ocr_text}\n\n"
            f"--- [STAGE 2: TEXT LLM JSON] ---\n{raw_content}"
        )

        parsed_dict = _extract_json_payload(raw_content)
        normalized_dict = _normalize_receipt_dict(parsed_dict)
        return ReceiptData(**normalized_dict)

    def _call_openai_compatible(self, request_kwargs: dict, target_model: str):
        try:
            return self.client.chat.completions.create(**request_kwargs)
        except AuthenticationError as e:
            raise ValueError(
                "Autentikasi OpenRouter gagal (401). Periksa kembali OPENROUTER_API_KEY Anda."
            ) from e
        except APIConnectionError as e:
            raise ConnectionError(
                f"Gagal terhubung ke server ({self.client.base_url}). Pastikan server lokal sudah berjalan."
            ) from e
        except NotFoundError as e:
            raise ValueError(
                f"Model '{target_model}' tidak ditemukan di endpoint {self.client.base_url}."
            ) from e

    def encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def parse(self, image_path: str) -> ReceiptData:
        if self.provider == "huggingface":
            if self.hf_arch == "donut":
                return self._run_donut(image_path)

            prompt_instruction = (
                "Read this receipt and output valid JSON with exact keys: "
                '{"merchant_name": str, "transaction_date": "YYYY-MM-DD", '
                '"total_amount": float, "line_items": [{"item_name": str, '
                '"item_quantity": float, "item_price": float, "item_total": float}]}. '
                "Use numbers only for prices."
            )
            raw_content = self._run_hf_vlm(image_path, prompt_instruction, json_prefill=True)
            self.last_raw_response = raw_content

            try:
                parsed_dict = _extract_json_payload(raw_content)
                normalized_dict = _normalize_receipt_dict(parsed_dict)
                return ReceiptData(**normalized_dict)
            except (json.JSONDecodeError, ValueError):
                return _fallback_parse_unstructured_text(raw_content)

        if self.provider == "hybrid":
            if self.hf_arch == "donut":
                donut_data = self._run_donut(image_path)
                ocr_text = donut_data.model_dump_json(indent=2)
            else:
                ocr_text = self._run_hf_vlm(
                    image_path,
                    "Transcribe all text in this receipt accurately line by line, including store name, date, items, quantities, and prices.",
                )
            if not ocr_text.strip():
                raise ValueError("Model OCR lokal tidak mendeteksi teks pada gambar struk.")
            return self._structure_text_with_llm(ocr_text)

        # Provider: OpenRouter atau Local Server (LM Studio / Ollama / vLLM)
        base64_image = self.encode_image(image_path)
        schema_instruction = (
            "Ekstrak informasi struk belanja ini menjadi JSON murni tanpa teks pengantar "
            "dengan struktur kunci persis berikut:\n"
            '{"merchant_name": "string", "merchant_address": "string atau null", '
            '"transaction_date": "YYYY-MM-DD atau null", "transaction_time": "HH:MM:SS atau null", '
            '"total_amount": 0.0, "line_items": [{"item_name": "string", "item_quantity": 1, '
            '"item_price": 0.0, "item_total": 0.0}]}\n'
            "Pastikan semua angka berupa angka murni tanpa simbol mata uang."
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": schema_instruction},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            }
        ]

        request_kwargs = {"model": self.model_name, "messages": messages}
        if self.provider == "openrouter":
            request_kwargs["response_format"] = {"type": "json_object"}

        response = self._call_openai_compatible(request_kwargs, self.model_name)
        raw_content = response.choices[0].message.content or "{}"
        self.last_raw_response = raw_content

        parsed_dict = _extract_json_payload(raw_content)
        normalized_dict = _normalize_receipt_dict(parsed_dict)
        return ReceiptData(**normalized_dict)