"""
llm_client.py — Groq only (free, fast)
Uses `requests` library to bypass Cloudflare 1010 bot block.
Get key at: https://console.groq.com
"""

import json

try:
    import requests
    USE_REQUESTS = True
except ImportError:
    USE_REQUESTS = False


class GroqClient:
    API_URL = "https://api.groq.com/openai/v1/chat/completions"
    MODEL   = "llama-3.3-70b-versatile"

    def __init__(self, api_key: str):
        if not api_key or not api_key.strip():
            raise ValueError("Groq API key is missing.")
        self.api_key = api_key.strip()

    def chat(self, system: str, user_message: str, max_tokens: int = 1500) -> str:
        payload = {
            "model":       self.MODEL,
            "max_tokens":  max_tokens,
            "temperature": 0.1,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user_message},
            ]
        }

        # Full browser-like headers — bypasses Cloudflare 1010 error
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type":  "application/json",
            "Accept":        "application/json",
            "User-Agent":    (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
        }

        print(f"[GROQ] Sending request...")

        if USE_REQUESTS:
            return self._call_requests(payload, headers)
        else:
            return self._call_urllib(payload, headers)

    # ── requests (preferred) ──────────────────────────────────────────────────
    def _call_requests(self, payload: dict, headers: dict) -> str:
        try:
            resp = requests.post(
                self.API_URL,
                json=payload,
                headers=headers,
                timeout=30
            )
            print(f"[GROQ] HTTP {resp.status_code}")

            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"]

            try:
                err = resp.json().get("error", {}).get("message", resp.text)
            except Exception:
                err = resp.text
            self._raise_for_status(resp.status_code, err)

        except requests.exceptions.SSLError as e:
            raise RuntimeError(
                "SSL error connecting to Groq.\n"
                "Fix: pip install --upgrade certifi"
            ) from e
        except requests.exceptions.ConnectionError as e:
            raise RuntimeError(
                "Cannot reach Groq API. Check your internet connection."
            ) from e
        except requests.exceptions.Timeout:
            raise RuntimeError("Groq request timed out. Try again.") from None

    # ── urllib fallback ───────────────────────────────────────────────────────
    def _call_urllib(self, payload: dict, headers: dict) -> str:
        import urllib.request, urllib.error
        data = json.dumps(payload).encode("utf-8")
        req  = urllib.request.Request(
            self.API_URL, data=data, headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                print(f"[GROQ] HTTP {resp.status}")
                result = json.loads(resp.read().decode("utf-8"))
                return result["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8")
            try:
                err = json.loads(body).get("error", {}).get("message", body)
            except Exception:
                err = body
            self._raise_for_status(e.code, err)
        except urllib.error.URLError as e:
            raise RuntimeError(f"Network error: {e.reason}") from e

    # ── error messages ────────────────────────────────────────────────────────
    def _raise_for_status(self, code: int, detail: str):
        messages = {
            401: "Invalid Groq API key. Get a valid one at https://console.groq.com",
            403: "Groq blocked the request (Cloudflare 1010). Run: pip install requests  then restart.",
            429: "Groq rate limit reached. Wait a moment and retry.",
            503: "Groq is temporarily unavailable. Try again shortly.",
        }
        msg = messages.get(code, f"Groq API error {code}: {detail}")
        raise RuntimeError(msg)