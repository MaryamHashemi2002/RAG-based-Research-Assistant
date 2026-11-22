import os
import sys
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
MODEL = os.getenv("LLM_MODEL", "llama3.2:1b")
API_KEY = os.getenv("LLM_API_KEY", "ollama")  # Ollama ignores the key

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = OpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=300)
    return _client


def generate(messages, temperature=0.2, max_tokens=500, client=None):
    client = client or _get_client()
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()


def ask(prompt, system=None, **kwargs):
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return generate(messages, **kwargs)


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "In two sentences, what is feature selection in machine learning?"
    print(f"Model: {MODEL} at {BASE_URL}")
    start = time.perf_counter()
    print(ask(question, system="You are a concise assistant."))
    print(f"\nTook {time.perf_counter() - start:.1f} s")