"""DeepSeek API wrapper: retry, concurrency, disk cache, cost stats."""
import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

CACHE_DIR = Path(os.environ.get("MBT_LLM_CACHE", str(Path(__file__).resolve().parents[1] / "outputs" / "llm_cache")))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-flash"

_client = None
_lock = threading.Lock()
_stats = {"calls": 0, "cache_hits": 0, "prompt_tokens": 0, "completion_tokens": 0}


def get_client() -> OpenAI:
    global _client
    if _client is None:
        with _lock:
            if _client is None:
                _client = OpenAI(
                    api_key=os.environ.get("OPENAI_API_KEY", ""),
                    base_url=os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL),
                    timeout=180.0,
                    max_retries=0,
                )
    return _client


def _cache_key(model: str, messages: list, kwargs: dict) -> str:
    blob = json.dumps({"m": model, "msg": messages, "kw": kwargs}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()


def chat(messages: list, model: str = None, max_attempts: int = 4, use_cache: bool = True, **kwargs) -> str:
    """Single chat completion with disk cache + exponential backoff. Returns content string."""
    model = model or os.environ.get("OPENAI_MODEL", DEFAULT_MODEL)
    key = _cache_key(model, messages, kwargs)
    cache_file = CACHE_DIR / f"{key}.json"
    if use_cache and cache_file.exists():
        _stats["cache_hits"] += 1
        return json.loads(cache_file.read_text())["content"]

    last_err = None
    for attempt in range(max_attempts):
        try:
            resp = get_client().chat.completions.create(model=model, messages=messages, **kwargs)
            content = resp.choices[0].message.content
            _stats["calls"] += 1
            if resp.usage:
                _stats["prompt_tokens"] += resp.usage.prompt_tokens or 0
                _stats["completion_tokens"] += resp.usage.completion_tokens or 0
            cache_file.write_text(json.dumps({"content": content}, ensure_ascii=False))
            return content
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(min(2 ** attempt * 2, 30))
    raise RuntimeError(f"DeepSeek call failed after {max_attempts} attempts: {last_err}")


def chat_batch(jobs: list, max_workers: int = 8, **kwargs) -> list:
    """jobs: list of messages-lists. Returns list of content strings (order preserved).
    Individual failures yield "" (scored as failure downstream) instead of
    aborting the whole batch; rerun refills them from the API/cache."""
    results = [None] * len(jobs)
    n_fail = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(chat, msgs, **kwargs): i for i, msgs in enumerate(jobs)}
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                results[i] = fut.result()
            except Exception as e:  # noqa: BLE001
                n_fail += 1
                print(f"  chat_batch: job {i} failed permanently ({e}); scored as empty",
                      flush=True)
                results[i] = ""
    if n_fail:
        print(f"  chat_batch: {n_fail}/{len(jobs)} jobs failed", flush=True)
    return results


def stats() -> dict:
    return dict(_stats)


def reset_stats():
    for k in _stats:
        _stats[k] = 0
