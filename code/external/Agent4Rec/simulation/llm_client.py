import os
import json
import urllib.error
import urllib.request

try:
    import requests
except ImportError:  # Fall back to stdlib HTTP client in lean environments.
    requests = None

try:
    import openai
except ImportError:  # Responses API path only needs requests.
    openai = None


def _extract_response_text(payload):
    output = payload.get("output")
    if isinstance(output, list):
        texts = []
        for item in output:
            content = item.get("content", [])
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") in {"output_text", "text"}:
                    text = block.get("text", "")
                    if text:
                        texts.append(text)
        if texts:
            return "\n".join(texts)
    return payload.get("output_text", "")


def _responses_base_url(openai_api_base):
    base = openai_api_base.rstrip("/")
    if base.endswith("/responses"):
        return base
    return f"{base}/responses"


def request_completion(
    *,
    messages,
    model,
    temperature,
    timeout,
    max_tokens,
    api_style,
):
    if api_style == "responses":
        api_key = os.getenv("OPENAI_API_KEY", "")
        openai_api_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
        url = _responses_base_url(openai_api_base)
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": os.getenv("OPENAI_API_USER_AGENT", "Mozilla/5.0"),
        }
        body = {
            "model": model,
            "input": messages,
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        if requests is not None:
            resp = requests.post(url, headers=headers, json=body, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
        else:
            req = urllib.request.Request(
                url,
                data=json.dumps(body).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:1000]
                raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
        text = _extract_response_text(data)
        usage = data.get("usage", {}) if isinstance(data, dict) else {}
        if usage and usage.get("total_tokens") is not None:
            k_tokens = usage["total_tokens"] / 1000.0
        else:
            # Rough fallback when provider does not return token usage.
            k_tokens = max(len(text) / 4000.0, 0.001)
        return {"content": text, "k_tokens": k_tokens, "raw": data}

    if openai is None:
        raise RuntimeError("openai package is required for chat_completions api_style")

    completion = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
        request_timeout=timeout,
        max_tokens=max_tokens,
    )
    usage = completion.get("usage", {})
    k_tokens = usage.get("total_tokens", 0) / 1000.0
    content = completion["choices"][0]["message"]["content"]
    return {"content": content, "k_tokens": k_tokens, "raw": completion}
