"""The website's pieces (app.py puts them together).

  resources.py   objects shared by every visitor: search model, AI clients, demo index, caches
  common.py      per-visitor state and helpers used by several tabs (slides, sources, traces)
  ask.py         Ask tab: chat with the slides
  practice.py    Practice exam tab
  browse.py      Browse lectures tab and the slide viewer
  compare.py     Compare models tab: the same question to several AI models, with votes
  games.py       mini games for a break (Citation Runner, Term Match), opened from the sidebar
  i18n.py        interface text in English, Russian and Korean: t("key") in the chosen language

Per-visitor state lives in st.session_state, never in module variables: these modules are
shared by everyone using the site.
"""
