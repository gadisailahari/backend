"""LLM client. Reads GROQ_API_KEY from the environment (set it in Railway Variables, never in code)."""
import json, os, time, urllib.request, urllib.error

MODEL = os.environ.get("LLM_MODEL", "openai/gpt-oss-120b")   # llama-3.3-70b-versatile was retired by Groq on 2026-08-16


def groq_llm(messages, json_mode=True, max_tokens=1200, temperature=0.2):
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY is not set in this environment.")
    body = {"model": MODEL, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    if MODEL.startswith("openai/gpt-oss"):                 # reasoning model: keep thinking short, leave room for the JSON answer
        body["reasoning_effort"] = "low"; body["max_tokens"] = max(max_tokens, 4000)
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    req = urllib.request.Request("https://api.groq.com/openai/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}",
                                          "User-Agent": "pulse-and-plate/1.0 (python)", "Accept": "application/json"}, method="POST")
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode())["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < 2:          # free-tier rate limit: wait and retry
                time.sleep(2 * (attempt + 1)); continue
            try: detail = e.read().decode()[:200]          # Groq's own reason (never contains your key)
            except Exception: detail = ""
            raise RuntimeError(f"LLM API error {e.code}: {detail}") from None
