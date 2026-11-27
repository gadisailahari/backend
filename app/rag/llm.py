import json, os, re, time, threading, urllib.request, urllib.error
LAST = threading.local()
MODEL = os.environ.get('LLM_MODEL', 'openai/gpt-oss-120b')

def groq_llm(messages, json_mode=True, max_tokens=1200, temperature=0.2):
    key = os.environ.get('GROQ_API_KEY')
    if not key:
        raise RuntimeError('GROQ_API_KEY is not set in this environment.')
    use_json_mode = json_mode
    for attempt in range(6):
        body = {'model': MODEL, 'messages': messages, 'temperature': temperature, 'max_tokens': max_tokens}
        if MODEL.startswith('openai/gpt-oss'):
            body['reasoning_effort'] = 'low'
            body['max_tokens'] = max(max_tokens, 2000)
        if use_json_mode:
            body['response_format'] = {'type': 'json_object'}
        req = urllib.request.Request('https://api.groq.com/openai/v1/chat/completions', data=json.dumps(body).encode(), headers={'Content-Type': 'application/json', 'Authorization': f'Bearer {key}', 'User-Agent': 'pulse-and-plate/1.0 (python)', 'Accept': 'application/json'}, method='POST')
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                data = json.loads(r.read().decode())
                LAST.meta = {'model': data.get('model'), 'system_fingerprint': data.get('system_fingerprint'), 'response_id': data.get('id'), 'usage': data.get('usage')}
                content = data['choices'][0]['message'].get('content')
                if not content:
                    raise RuntimeError('LLM returned an empty answer')
                return content
        except urllib.error.HTTPError as e:
            try:
                raw = e.read().decode()
            except Exception:
                raw = ''
            if e.code == 429 and attempt < 5:
                m = re.search('try again in ([0-9.]+)\\s*(ms|s)', raw)
                wait = float(m.group(1)) / (1000 if m.group(2) == 'ms' else 1) + 1.0 if m else 8.0
                time.sleep(min(max(wait, 2.0), 25.0))
                continue
            if e.code == 400 and 'json_validate_failed' in raw:
                if use_json_mode:
                    use_json_mode = False
                    continue
                try:
                    draft = json.loads(raw)['error'].get('failed_generation')
                    if draft:
                        return draft
                except Exception:
                    pass
            raise RuntimeError(f'LLM API error {e.code}: {raw[:200]}') from None
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            if attempt < 2:
                time.sleep(2)
                continue
            raise RuntimeError(f'LLM network error: {type(e).__name__}') from None
    raise RuntimeError('LLM call failed after retries')
