"""Small OpenAI-compatible client: persistent cache, retries, no SDK required."""
import hashlib
import json
import os
import random
import threading
import urllib.error
import urllib.request
import uuid
from pathlib import Path


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def safe_error(error):
    text = str(error)
    for name, value in os.environ.items():
        if value and len(value) > 6 and any(word in name.upper() for word in ("KEY", "TOKEN", "SECRET")):
            text = text.replace(value, "[REDACTED]")
    return text[:500]


class Client:
    def __init__(self, config, cache_dir, stop=None):
        self.config = config
        self.cache_dir = Path(cache_dir)
        self.stop = stop if stop is not None else threading.Event()
        self.guard, self.locks = threading.Lock(), {}

    def forget(self, response):
        # A syntactically invalid judge verdict must not poison future retries.
        key = response["request_id"]
        if len(key) == 64 and all(c in "0123456789abcdef" for c in key):
            (self.cache_dir / (key + ".json")).unlink(missing_ok=True)

    def call(self, prompt):
        config = self.config
        parameters = config.get("parameters", {})
        if set(parameters) & {"messages", "model", "stream", "tools", "tool_choice"}:
            raise ValueError("parameters must not override messages/model/stream or add search tools")
        endpoint = config["base_url"].rstrip("/") + "/chat/completions"
        payload = {**parameters, "model": config["model"],
                   "messages": [{"role": "user", "content": prompt}], "stream": False}
        key = fingerprint({"endpoint": endpoint, "payload": payload})
        with self.guard:
            lock = self.locks.setdefault(key, threading.Lock())
        with lock:
            path = self.cache_dir / (key + ".json")
            if path.exists():
                return {**json.loads(path.read_text(encoding="utf-8")), "cached": True}
            if self.stop.is_set():
                raise RuntimeError("Stopped after a fatal API error; fix the key/account and resume")
            token = os.environ.get(config["api_key_env"])
            if not token:
                self.stop.set()
                raise ValueError(f"Set environment variable {config['api_key_env']}")
            request = urllib.request.Request(endpoint, data=json.dumps(payload).encode(), headers={
                "Authorization": f"Bearer {token}", "Content-Type": "application/json"})
            attempts = int(config.get("attempts", 5))
            for attempt in range(attempts):
                try:
                    with urllib.request.urlopen(request, timeout=config.get("timeout", 600)) as response:
                        body = json.load(response)
                    if body.get("error"):
                        raise ValueError("API returned an error object: " + str(body["error"]))
                    choice = body["choices"][0]
                    text = choice["message"].get("content")
                    if choice.get("finish_reason") != "stop" or not isinstance(text, str) or not text.strip():
                        raise ValueError(f"Empty or incomplete response (finish_reason={choice.get('finish_reason')})")
                    result = {"request_id": key, "model": body.get("model", config["model"]),
                              "text": text.strip(), "usage": body.get("usage"), "cached": False}
                    save_json(path, result)
                    return result
                except urllib.error.HTTPError as error:
                    code = error.code
                    # Do not echo raw provider responses: they may contain secrets or huge prompts.
                    error.close()
                    if code in (401, 402, 403):
                        self.stop.set()
                    if code not in (408, 429, 500, 502, 503, 504) or attempt + 1 == attempts:
                        raise RuntimeError(f"HTTP {code} from {config['base_url']}") from None
                except (urllib.error.URLError, TimeoutError, ConnectionError):
                    if attempt + 1 == attempts:
                        raise RuntimeError("API connection failed after retries") from None
                if self.stop.wait(min(30, 2 ** attempt + random.random())):
                    raise RuntimeError("Stopped after a fatal API error")
            raise RuntimeError("No API attempt made; attempts must be positive")
