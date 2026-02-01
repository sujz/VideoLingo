import os
import json
from threading import Lock
import json_repair
from core.utils.config_utils import load_key
from rich import print as rprint
from core.utils.decorator import except_handler

try:
    import httpx
except Exception:
    httpx = None
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# ------------
# cache gpt response
# ------------

LOCK = Lock()
GPT_LOG_FOLDER = 'output/gpt_log'

def _save_cache(model, prompt, resp_content, resp_type, resp, message=None, log_title="default"):
    with LOCK:
        logs = []
        file = os.path.join(GPT_LOG_FOLDER, f"{log_title}.json")
        os.makedirs(os.path.dirname(file), exist_ok=True)
        if os.path.exists(file):
            with open(file, 'r', encoding='utf-8') as f:
                logs = json.load(f)
        logs.append({"model": model, "prompt": prompt, "resp_content": resp_content, "resp_type": resp_type, "resp": resp, "message": message})
        with open(file, 'w', encoding='utf-8') as f:
            json.dump(logs, f, ensure_ascii=False, indent=4)

def _load_cache(prompt, resp_type, log_title):
    with LOCK:
        file = os.path.join(GPT_LOG_FOLDER, f"{log_title}.json")
        if os.path.exists(file):
            with open(file, 'r', encoding='utf-8') as f:
                for item in json.load(f):
                    if item["prompt"] == prompt and item["resp_type"] == resp_type:
                        return item["resp"]
        return False

# ------------
# ask gpt once
# ------------

@except_handler("GPT request failed", retry=2)
def ask_gpt(prompt, resp_type=None, valid_def=None, log_title="default"):
    provider = load_key("api.provider") if load_key("api.provider") else None
    base_url = load_key("api.base_url") if load_key("api.base_url") else None
    # If provider is ollama, we allow local calls without API key
    if provider != 'ollama' and not load_key("api.key"):
        raise ValueError("API key is not set")
    # check cache
    cached = _load_cache(prompt, resp_type, log_title)
    if cached:
        rprint("use cache response")
        return cached

    model = load_key("api.model")
    base_url = load_key("api.base_url")

    # Ollama local provider support
    if provider == 'ollama' or (base_url and ('localhost' in base_url or 'ollama' in str(base_url))):
        if httpx is None:
            raise RuntimeError("httpx is required for Ollama support")
        ollama_url = base_url or 'http://127.0.0.1:11434'
        # Ollama generate endpoint
        gen_url = ollama_url.rstrip('/') + '/api/generate'
        payload = {
            'model': model,
            'prompt': prompt,
            'max_tokens': 2048,
            'temperature': 0.2,
        }
        r = httpx.post(gen_url, json=payload, timeout=300.0)
        if r.status_code != 200:
            raise RuntimeError(f"Ollama request failed: {r.status_code} {r.text}")
        # Save raw response for debugging
        try:
            os.makedirs(GPT_LOG_FOLDER, exist_ok=True)
            raw_file = os.path.join(GPT_LOG_FOLDER, f"{log_title}_raw.txt")
            with open(raw_file, 'w', encoding='utf-8') as rf:
                rf.write(r.text)
        except Exception:
            pass
        # Try to parse JSON. Ollama/adapters sometimes return multiple JSON objects
        # (NDJSON or streaming) or non-standard shapes. Handle gracefully.
        parsed_items = []
        try:
            j = r.json()
            parsed_items = [j]
        except Exception:
            # Fallback: try to parse each line as JSON (NDJSON style)
            for line in r.text.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    parsed_items.append(json.loads(line))
                except Exception:
                    # ignore non-json lines
                    continue

        # Extract textual content from parsed items.
        # Many Ollama adapters stream objects with a `response` field per-chunk.
        resp_content = ''
        if parsed_items:
            parts = []
            for item in parsed_items:
                if not isinstance(item, dict):
                    continue
                if 'response' in item:
                    parts.append(item.get('response') or '')
                elif 'generated' in item:
                    parts.append(item.get('generated') or '')
                elif 'choices' in item and len(item['choices']) > 0:
                    c = item['choices'][0]
                    parts.append(c.get('text') or c.get('message') or '')
                elif 'text' in item:
                    parts.append(item.get('text') or '')
            resp_content = ''.join(parts).strip()
            if not resp_content:
                resp_content = r.text
            # If expecting JSON, extract JSON block from concatenated response
            if resp_type == "json":
                content_to_parse = resp_content
                try:
                    # Prefer ```json ... ``` blocks
                    if '```json' in content_to_parse:
                        start = content_to_parse.find('```json') + len('```json')
                        end = content_to_parse.rfind('```')
                        if end > start:
                            content_to_parse = content_to_parse[start:end].strip()
                    else:
                        # Fallback: try to extract the first {...} ... matching braces
                        first_brace = content_to_parse.find('{')
                        last_brace = content_to_parse.rfind('}')
                        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                            content_to_parse = content_to_parse[first_brace:last_brace+1]
                except Exception:
                    # keep original if extraction fails
                    content_to_parse = resp_content
                try:
                    resp = json_repair.loads(content_to_parse)
                    if not isinstance(resp, dict):
                        raise ValueError(f"LLM output is not a JSON object: {resp}")
                except Exception as e:
                    raise RuntimeError(f"Failed to parse LLM JSON output: {e}\nRaw: {content_to_parse}")
            else:
                resp = resp_content
        else:
            resp_content = r.text
    else:
        # OpenAI-compatible path
        if OpenAI is None:
            raise RuntimeError("OpenAI package is not installed or unavailable")
        if 'ark' in base_url:
            base_url = "https://ark.cn-beijing.volces.com/api/v3" # huoshan base url
        elif 'v1' not in base_url:
            base_url = base_url.strip('/') + '/v1'
        client = OpenAI(api_key=load_key("api.key"), base_url=base_url)
        response_format = {"type": "json_object"} if resp_type == "json" and load_key("api.llm_support_json") else None

        messages = [{"role": "user", "content": prompt}]

        params = dict(
            model=model,
            messages=messages,
            response_format=response_format,
            timeout=300
        )
        resp_raw = client.chat.completions.create(**params)

        # Save raw OpenAI-compatible response for debugging
        try:
            os.makedirs(GPT_LOG_FOLDER, exist_ok=True)
            raw_file = os.path.join(GPT_LOG_FOLDER, f"{log_title}_raw.txt")
            with open(raw_file, 'w', encoding='utf-8') as rf:
                rf.write(str(resp_raw))
        except Exception:
            pass

        # process and return full result
        resp_content = resp_raw.choices[0].message.content
    # (OpenAI path already handled above)
    
    # check if the response format is valid
    if valid_def:
        valid_resp = valid_def(resp)
        if valid_resp['status'] != 'success':
            _save_cache(model, prompt, resp_content, resp_type, resp, log_title="error", message=valid_resp['message'])
            raise ValueError(f"❎ API response error: {valid_resp['message']}")

    _save_cache(model, prompt, resp_content, resp_type, resp, log_title=log_title)
    return resp


if __name__ == '__main__':
    from rich import print as rprint
    
    result = ask_gpt("""test respond ```json\n{\"code\": 200, \"message\": \"success\"}\n```""", resp_type="json")
    rprint(f"Test json output result: {result}")
