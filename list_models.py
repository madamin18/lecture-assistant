"""Print the chat models your key can use.  Run: python list_models.py"""
from openai import OpenAI
import config

client = OpenAI(api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL)
for m in client.models.list():
    print(m.id)
