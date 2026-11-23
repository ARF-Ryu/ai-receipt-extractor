import traceback

import streamlit as st
from dotenv import load_dotenv

from image_processor import preprocess_receipt
from parser import AIReceiptParser

load_dotenv("secrets.env")
load_dotenv()

st.set_page_config(page_title="AI Receipt Extractor", page_icon="🧾", layout="wide")


@st.cache_resource(show_spinner="Memuat model AI ke memori...")
def get_cached_parser(
    provider: str,
    model_name: str,
    base_url: str = None,
    api_key: str = None,
    llm_backend: str = "openrouter",
    llm_model_name: str = None,
) -> AIReceiptParser:
    return AIReceiptParser(
        provider=provider,
        model_name=model_name,
        base_url=base_url,
        api_key=api_key,
        llm_backend=llm_backend,
        llm_model_name=llm_model_name,
    )


st.title("🧾 AI Receipt Extractor & Parser (OpenCV + Multi-AI)")
st.write(
    "Sistem ekstraksi data finansial otomatis menggunakan Computer Vision, OpenRouter, "
    "Server Lokal (LM Studio / Ollama), dan Hugging Face Transformers."
)

st.sidebar.header("⚙️ Konfigurasi Pipeline")
provider_choice = st.sidebar.selectbox(
    "Pilih Arsitektur / Provider",
    [
        "OpenRouter (Vision Cloud)",
        "Hugging Face (SmolVLM / Donut)",
        "Local Server (LM Studio / Ollama)",
        "Hybrid (HF Local OCR + Text LLM)",
    ],
)
enable_opencv = st.sidebar.checkbox("Gunakan OpenCV Preprocessing", value=True)

base_url = None
api_key_input = None
llm_backend = "openrouter"
llm_model_name = None

if "OpenRouter" in provider_choice:
    provider = "openrouter"
    model_name = st.sidebar.text_input("Model Vision OpenRouter", value="openai/gpt-4o-mini")
    api_key_input = st.sidebar.text_input(
        "OpenRouter API Key (Opsional)",
        type="password",
        help="Kosongkan jika sudah diisi di config.py atau secrets.env",
    )

elif "Hugging Face" in provider_choice:
    provider = "huggingface"
    model_name = st.sidebar.selectbox(
        "Model Hugging Face",
        [
            "HuggingFaceTB/SmolVLM-500M-Instruct",
            "naver-clova-ix/donut-base-finetuned-cord-v2",
            "Qwen/qwen2.5vl:3b",
        ],
    )
    st.sidebar.info(
        "SmolVLM-500M ringan untuk CPU/GPU dan langsung menghasilkan JSON. "
        "Donut CORD-v2 khusus untuk ekstraksi item & total struk."
    )

elif "Local Server" in provider_choice:
    provider = "local"
    base_url = st.sidebar.text_input(
        "Local Server Base URL",
        value="http://localhost:11434/v1",
        help="LM Studio: http://localhost:1234/v1 | Ollama: http://localhost:11434/v1 | vLLM: http://localhost:8000/v1",
    )
    model_name = st.sidebar.text_input("Nama Model Vision Lokal", value="qwen2.5vl:3b")

    st.sidebar.info(
        "Uncheklist OpenCv Processing jika menggunakan pilihan provider ini"
    )

else:
    provider = "hybrid"
    model_name = st.sidebar.selectbox(
        "Stage 1: Model Lokal (HF)",
        [
            "HuggingFaceTB/SmolVLM-500M-Instruct",
            "naver-clova-ix/donut-base-finetuned-cord-v2",
        ],
    )
    stage2_choice = st.sidebar.selectbox(
        "Stage 2: Backend Text LLM",
        ["OpenRouter", "Local Server (LM Studio / Ollama)"],
    )
    if stage2_choice == "OpenRouter":
        llm_backend = "openrouter"
        llm_model_name = st.sidebar.text_input("Stage 2: Model Teks OpenRouter", value="openai/gpt-4o-mini")
        api_key_input = st.sidebar.text_input(
            "OpenRouter API Key (Opsional)",
            type="password",
            help="Kosongkan jika sudah diisi di config.py atau secrets.env",
        )
    else:
        llm_backend = "local"
        base_url = st.sidebar.text_input("Local Server Base URL", value="http://localhost:1234/v1")
        llm_model_name = st.sidebar.text_input("Stage 2: Nama Model Lokal", value="qwen2.5-3b-instruct")

uploaded_file = st.file_uploader("Unggah foto struk belanja (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    temp_path = "temp_original.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    col1, col2 = st.columns(2)
    with col1:
        st.image(temp_path, caption="Struk Asli", use_container_width=True)

    target_path = temp_path
    if enable_opencv:
        processed_path = "temp_processed.jpg"
        target_path = preprocess_receipt(temp_path, processed_path)
        with col2:
            st.image(processed_path, caption="Hasil Preprocessing (OpenCV)", use_container_width=True)

    if st.button("🚀 Jalankan Ekstraksi AI", type="primary"):
        with st.spinner("Menjalankan ekstraksi AI..."):
            parser = None
            try:
                parser = get_cached_parser(
                    provider=provider,
                    model_name=model_name,
                    base_url=base_url,
                    api_key=api_key_input or None,
                    llm_backend=llm_backend,
                    llm_model_name=llm_model_name,
                )
                result = parser.parse(target_path)

                st.success("Ekstraksi Berhasil!")

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Merchant", result.merchant_name)
                c2.metric("Tanggal", result.transaction_date or "-")
                c3.metric("Waktu", result.transaction_time or "-")
                c4.metric("Total Pembayaran", f"Rp {result.total_amount:,.2f}")

                st.subheader("Rincian Belanja (Line Items)")
                st.table([item.model_dump() for item in result.line_items])

                if parser.last_raw_response:
                    with st.expander("Lihat Output Mentah Model (Debug)"):
                        st.code(parser.last_raw_response)

                st.download_button(
                    label="📥 Unduh JSON Terstruktur",
                    data=result.model_dump_json(indent=2),
                    file_name="receipt_extracted.json",
                    mime="application/json",
                )
            except Exception as e:
                st.error(f"Gagal memproses struk: {e}")
                if parser and parser.last_raw_response:
                    with st.expander("Output Mentah Model Sebelum Error"):
                        st.code(parser.last_raw_response)
                with st.expander("Detail Traceback Error"):
                    st.code(traceback.format_exc())