"""Check every AI provider that has a key: which models it lists, and whether the models in the
"Auto" chain actually answer. Run after adding a key:  python check_models.py
(Uses one tiny request per Auto model.)"""
import time

import config
from rag import auto_chain, list_models, make_clients, model_label, split_ref, strip_thinking

clients = make_clients()
print("Providers with a key:", ", ".join(clients) or "none (add GEMINI_API_KEY or GROQ_API_KEY to .env)")
for provider, client in clients.items():
    models = list_models(provider, client)
    print(f"\n{config.PROVIDERS[provider]['label']} ({config.PROVIDERS[provider]['base_url']}): "
          f"{len(models)} models in the picker")
    for m in models:
        print("   ", m)

print("\nAuto chain (tried in this order):")
for ref in auto_chain():
    provider, name = split_ref(ref)
    if provider not in clients:
        continue
    t0 = time.time()
    try:
        r = clients[provider].chat.completions.create(
            model=name, messages=[{"role": "user", "content": "Reply with just: OK"}])
        reply = strip_thinking(r.choices[0].message.content or "")
        print(f"  WORKS   {model_label(ref)}  ({time.time() - t0:.1f} s): {reply[:30]!r}")
    except Exception as e:  # noqa: BLE001  (show any failure)
        print(f"  FAILED  {model_label(ref)}: {type(e).__name__} {str(e)[:150]}")
