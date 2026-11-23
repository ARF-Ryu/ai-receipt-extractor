import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--provider",
        action="store",
        default="openrouter",
        help="Pilih provider yang diuji: openrouter, huggingface, local, hybrid, atau all",
    )
    parser.addoption(
        "--opencv",
        action="store_true",
        default=False,
        help="Aktifkan preprocessing OpenCV sebelum ekstraksi",
    )
    parser.addoption(
        "--openrouter-model",
        action="store",
        default="openai/gpt-4o-mini",
        help="Nama model vision untuk provider OpenRouter",
    )
    parser.addoption(
        "--hf-model",
        action="store",
        default="HuggingFaceTB/SmolVLM-500M-Instruct",
        help="Nama model Hugging Face (SmolVLM atau Donut CORD-v2)",
    )
    parser.addoption(
        "--local-url",
        action="store",
        default="http://localhost:11434/v1",
        help="Base URL server lokal (Ollama: :11434/v1 | LM Studio: :1234/v1)",
    )
    parser.addoption(
        "--local-model",
        action="store",
        default="qwen2.5vl:3b",
        help="Nama model vision pada server lokal",
    )
    parser.addoption(
        "--hybrid-backend",
        action="store",
        default="openrouter",
        help="Backend Stage 2 untuk mode hybrid: openrouter atau local",
    )
    parser.addoption(
        "--hybrid-llm-model",
        action="store",
        default="openai/gpt-4o-mini",
        help="Nama model Text LLM Stage 2 untuk mode hybrid",
    )