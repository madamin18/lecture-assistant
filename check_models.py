"""Try a few chat models with your key and show which ones actually answer.
Run: python check_models.py
"""
from openai import OpenAI
import config

CANDIDATES = ["gemini-flash-latest", "gemini-3.5-flash", "gemini-3.8-flash",
              "gemini-2.5-flash", "gemini-3.1-flash-lite", "gemini-flash-lite-latest"]

client = OpenAI(api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL)
print("base_url:", config.LLM_BASE_URL)
for model in CANDIDATES:
    try:
        r = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": "Reply with just: OK"}])
        print(f"WORKS   {model}: {r.choices[0].message.content.strip()[:30]}")
    except Exception as e:
        print(f"FAILED  {model}: {type(e).__name__} {str(e)[:150]}")
