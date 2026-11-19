"""Mini games for a short break ("Bored? Play a game" in the sidebar). Both are original and
use the course's vocabulary:

  Citation Runner  an endless runner: jump over HTTP error blocks (404, 429, 503), stay under
                   the hallucination ghosts, collect citation badges for bonus points
  Term Match       a memory game: flip cards to match each course term with its meaning

Each game is a small custom component (st.components.v2: HTML + canvas/JS, no iframe, no
outside files), coloured with the app's theme variables so it fits light and dark mode. When a
round ends it reports the result to Python, which keeps the visitor's best and the best on
this server."""
import threading

import streamlit as st

from ui import i18n
from ui.i18n import t

# ---------------------------------------------------------------- Citation Runner
RUNNER_HTML = """<div class="runner"><canvas aria-label="Citation Runner game"></canvas></div>"""

RUNNER_CSS = """
.runner { width: 100%; }
.runner canvas { display: block; width: 100%; touch-action: manipulation; outline: none;
                 border-radius: var(--st-base-radius); cursor: pointer; }
.runner canvas:focus-visible { box-shadow: 0 0 0 2px var(--st-primary-color); }
"""

RUNNER_JS = """
const games = new WeakMap()

function makeRunner(root) {
  const canvas = root.querySelector("canvas")
  const ctx = canvas.getContext("2d")
  const host = root.host ?? root
  const H = 230, GROUND = H - 34, GRAVITY = 2200, JUMP = 760
  let W = 640, text = {}, best = 0, send = null
  let state = "ready", raf = 0, last = performance.now()
  let p, obstacles, coins, speed, distance, bonus, nextSpawn, nextCoin, pops

  const css = (name, fallback) => getComputedStyle(host).getPropertyValue(name).trim() || fallback

  function resize() {
    const dpr = window.devicePixelRatio || 1
    W = Math.max(280, Math.min(canvas.parentElement.clientWidth || 640, 900))
    canvas.width = W * dpr
    canvas.height = H * dpr
    canvas.style.height = H + "px"
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
  }

  function reset() {
    p = { x: 56, y: GROUND - 36, w: 30, h: 36, vy: 0, onGround: true }
    obstacles = []; coins = []; pops = []
    speed = 320; distance = 0; bonus = 0; nextSpawn = 500; nextCoin = 900
  }

  const score = () => Math.floor(distance / 10) + bonus

  function jump() {
    if (state !== "running") { reset(); state = "running"; return }
    if (p.onGround) { p.vy = -JUMP; p.onGround = false }
  }

  function spawn() {
    if (Math.random() < 0.72 || score() < 120) {  // an error on the ground: jump over it
      const codes = ["404", "429", "500", "503"]
      const h = 26 + Math.random() * 22, w = 28 + Math.random() * 16
      obstacles.push({ kind: "block", x: W + 20, y: GROUND - h, w, h,
                       label: codes[Math.floor(Math.random() * codes.length)] })
    } else {  // a hallucination at head height: stay on the ground and run under it
      obstacles.push({ kind: "ghost", x: W + 20, y: GROUND - 84 - Math.random() * 12, w: 34, h: 28 })
    }
  }

  function hit(a, b, pad) {
    return a.x + pad < b.x + b.w && a.x + a.w - pad > b.x && a.y + pad < b.y + b.h && a.y + a.h - pad > b.y
  }

  function update(dt) {
    speed = Math.min(760, speed + 9 * dt)
    distance += speed * dt
    p.vy += GRAVITY * dt
    p.y += p.vy * dt
    if (p.y >= GROUND - p.h) { p.y = GROUND - p.h; p.vy = 0; p.onGround = true }

    nextSpawn -= speed * dt
    if (nextSpawn <= 0) { spawn(); nextSpawn = speed * (0.85 + Math.random() * 0.9) + 140 }
    nextCoin -= speed * dt
    if (nextCoin <= 0) {
      const ghostNear = obstacles.some(o => o.kind === "ghost" && o.x > W - 140)
      if (!ghostNear) coins.push({ x: W + 20, y: GROUND - 108, w: 40, h: 20,
                                   label: "L0" + (1 + Math.floor(Math.random() * 9)) })
      nextCoin = speed * (2.0 + Math.random() * 2.5)
    }
    for (const o of obstacles) o.x -= speed * dt
    for (const c of coins) c.x -= speed * dt
    for (const q of pops) { q.t -= dt; q.y -= 40 * dt }
    obstacles = obstacles.filter(o => o.x + o.w > -10)
    coins = coins.filter(c => c.x + c.w > -10 && !c.taken)
    pops = pops.filter(q => q.t > 0)

    for (const c of coins) {
      if (!c.taken && hit(p, c, 0)) { c.taken = true; bonus += 25; pops.push({ x: c.x, y: c.y, t: 0.7 }) }
    }
    if (obstacles.some(o => hit(p, o, 6))) {
      state = "over"
      best = Math.max(best, score())
      if (send) send("over", { score: score() })
    }
  }

  function rounded(x, y, w, h, r) {
    ctx.beginPath()
    ctx.roundRect ? ctx.roundRect(x, y, w, h, r) : ctx.rect(x, y, w, h)
  }

  function draw() {
    const fg = css("--st-text-color", "#1e1b4b"), bg = css("--st-background-color", "#ffffff")
    const panel = css("--st-secondary-background-color", "#f5f3ff")
    const primary = css("--st-primary-color", "#6d28d9"), border = css("--st-border-color", "#d4d4d8")
    const red = css("--st-red-color", "#dc2626"), green = css("--st-green-color", "#16a34a")
    const violet = css("--st-violet-color", "#8b5cf6"), font = css("--st-font", "sans-serif")
    ctx.clearRect(0, 0, W, H)
    ctx.fillStyle = panel; rounded(0, 0, W, H, 14); ctx.fill()

    // ground with moving dashes
    ctx.fillStyle = border
    ctx.fillRect(0, GROUND, W, 2)
    const off = (distance || 0) % 46
    for (let x = -off; x < W; x += 46) ctx.fillRect(x, GROUND + 9, 18, 2)

    // citation badges
    ctx.textAlign = "center"; ctx.textBaseline = "middle"
    for (const c of coins) {
      ctx.fillStyle = green; rounded(c.x, c.y, c.w, c.h, 6); ctx.fill()
      ctx.fillStyle = bg; ctx.font = `700 11px ${font}`; ctx.fillText(c.label, c.x + c.w / 2, c.y + c.h / 2 + 1)
    }
    // obstacles
    for (const o of obstacles) {
      if (o.kind === "block") {
        ctx.fillStyle = red; rounded(o.x, o.y, o.w, o.h, 5); ctx.fill()
        ctx.fillStyle = bg; ctx.font = `700 11px ${font}`; ctx.fillText(o.label, o.x + o.w / 2, o.y + Math.min(o.h / 2, 12) + 1)
      } else {
        const wob = Math.sin((distance || 0) / 30) * 2
        ctx.globalAlpha = 0.85; ctx.fillStyle = violet
        ctx.beginPath()
        ctx.arc(o.x + o.w / 2, o.y + 12 + wob, o.w / 2, Math.PI, 0)
        ctx.lineTo(o.x + o.w, o.y + o.h + wob)
        for (let i = 3; i >= 0; i--) ctx.lineTo(o.x + (o.w / 4) * i, o.y + o.h + wob - (i % 2 ? 5 : 0))
        ctx.closePath(); ctx.fill(); ctx.globalAlpha = 1
        ctx.fillStyle = bg
        ctx.beginPath(); ctx.arc(o.x + 11, o.y + 12 + wob, 3.2, 0, 7); ctx.arc(o.x + 23, o.y + 12 + wob, 3.2, 0, 7); ctx.fill()
      }
    }
    // the robot
    const r = p || { x: 56, y: GROUND - 36, w: 30, h: 36, onGround: true }
    ctx.fillStyle = primary
    ctx.fillRect(r.x + 14, r.y - 8, 2, 8)
    ctx.beginPath(); ctx.arc(r.x + 15, r.y - 9, 3, 0, 7); ctx.fill()
    rounded(r.x, r.y, r.w, r.h - 8, 7); ctx.fill()
    const step = r.onGround && state === "running" ? Math.floor((distance || 0) / 40) % 2 : 0
    ctx.fillRect(r.x + 5, r.y + r.h - 9, 7, 9 - step * 4)
    ctx.fillRect(r.x + 18, r.y + r.h - 9, 7, 5 + step * 4)
    ctx.fillStyle = bg; rounded(r.x + 5, r.y + 6, 20, 10, 4); ctx.fill()
    ctx.fillStyle = fg
    ctx.fillRect(r.x + 10, r.y + 9, 3, 4); ctx.fillRect(r.x + 18, r.y + 9, 3, 4)

    // +25 pops
    ctx.font = `700 13px ${font}`; ctx.fillStyle = green
    for (const q of pops) { ctx.globalAlpha = Math.max(0, q.t / 0.7); ctx.fillText("+25", q.x + 20, q.y); ctx.globalAlpha = 1 }

    // score
    ctx.fillStyle = fg; ctx.textAlign = "right"; ctx.font = `600 14px ${font}`
    ctx.fillText(`${text.score ?? "Score"} ${p ? score() : 0}   ${text.best ?? "Best"} ${best}`, W - 14, 20)

    // messages
    if (state !== "running") {
      ctx.globalAlpha = 0.88; ctx.fillStyle = panel  // keeps the text readable over the track
      rounded(W / 2 - Math.min(230, W / 2 - 10), 52, Math.min(460, W - 20), 72, 12); ctx.fill()
      ctx.globalAlpha = 1
      ctx.textAlign = "center"
      ctx.fillStyle = fg
      if (state === "over") {
        ctx.font = `700 22px ${font}`; ctx.fillText(`${text.over ?? "Game over"}: ${score()}`, W / 2, 78)
        ctx.font = `500 14px ${font}`; ctx.fillText(text.restart ?? "", W / 2, 106)
      } else {
        ctx.font = `600 16px ${font}`; ctx.fillText(text.start ?? "", W / 2, 84)
      }
    }
  }

  function frame(now) {
    const dt = Math.min(0.05, (now - last) / 1000)
    last = now
    if (state === "running") update(dt)
    draw()
    raf = requestAnimationFrame(frame)
  }

  canvas.tabIndex = 0
  canvas.onkeydown = (e) => {
    if ([" ", "ArrowUp", "w", "W", "Enter"].includes(e.key)) { e.preventDefault(); jump() }
  }
  canvas.onpointerdown = (e) => { e.preventDefault(); canvas.focus(); jump() }
  const ro = new ResizeObserver(() => resize())
  ro.observe(canvas.parentElement)
  reset()  // the start screen draws the robot and the (empty) track
  resize()
  raf = requestAnimationFrame(frame)

  return {
    update(data, setTriggerValue) {
      text = data?.text ?? {}
      best = Math.max(best, data?.best ?? 0)
      send = setTriggerValue
    },
    destroy() { cancelAnimationFrame(raf); ro.disconnect(); canvas.onkeydown = canvas.onpointerdown = null },
  }
}

export default function (component) {
  const { data, parentElement, setTriggerValue } = component
  let game = games.get(parentElement)
  if (!game) { game = makeRunner(parentElement); games.set(parentElement, game) }
  game.update(data, setTriggerValue)
  return () => { game.destroy(); games.delete(parentElement) }
}
"""

# ---------------------------------------------------------------- Term Match
MATCH_HTML = """<div class="match">
  <div class="bar"><span class="moves"></span><span class="time"></span><span class="msg"></span>
    <button type="button" class="new"></button></div>
  <div class="grid"></div>
</div>"""

MATCH_CSS = """
.match { font-family: var(--st-font); color: var(--st-text-color); }
.bar { display: flex; flex-wrap: wrap; gap: 14px; align-items: center; margin-bottom: 10px; }
.msg { font-weight: 700; color: var(--st-green-text-color); }
.new { margin-left: auto; font: inherit; padding: 4px 14px; cursor: pointer;
       border-radius: var(--st-button-radius); border: 1px solid var(--st-primary-color);
       background: transparent; color: var(--st-primary-color); }
.grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 8px; }
.card { min-height: 88px; padding: 8px; font: inherit; font-size: 0.8rem; line-height: 1.25;
        border-radius: var(--st-base-radius); border: 1px solid var(--st-primary-color);
        background: var(--st-primary-color); color: var(--st-background-color); cursor: pointer;
        transition: transform 0.15s; }
.card:hover { transform: translateY(-2px); }
.card .front { display: none; }
.card .back { font-size: 1.6rem; font-weight: 700; }
.card.open, .card.matched { background: var(--st-secondary-background-color);
                            color: var(--st-text-color); border-color: var(--st-border-color); }
.card.open .front, .card.matched .front { display: block; }
.card.open .back, .card.matched .back { display: none; }
.card.term .front { font-weight: 700; font-size: 0.9rem; }
.card.matched { background: var(--st-green-background-color); color: var(--st-green-text-color);
                border-color: var(--st-green-color); cursor: default; }
@media (max-width: 520px) { .card { min-height: 76px; font-size: 0.7rem; padding: 5px; } }
"""

MATCH_JS = """
const games = new WeakMap()

function makeMatch(root) {
  const grid = root.querySelector(".grid"), movesEl = root.querySelector(".moves")
  const timeEl = root.querySelector(".time"), msg = root.querySelector(".msg")
  const newBtn = root.querySelector(".new")
  let pool = [], poolKey = "", text = {}, send = null
  let cards = [], open = [], moves = 0, found = 0, started = 0, timer = 0, lock = false, done = false

  const shuffle = (a) => { for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [a[i], a[j]] = [a[j], a[i]] } return a }
  const seconds = () => (started ? Math.floor(((done || Date.now()) - started) / 1000) : 0)

  function status() {
    movesEl.textContent = `${text.moves ?? "Moves"}: ${moves}`
    timeEl.textContent = `${text.time ?? "Time"}: ${seconds()} s`
    msg.textContent = done ? (text.won ?? "") : ""
    newBtn.textContent = text.new ?? "New game"
  }

  function newGame() {
    const pairs = shuffle(pool.slice()).slice(0, 8)
    cards = shuffle(pairs.flatMap((p, i) => [{ pair: i, kind: "term", text: p[0] }, { pair: i, kind: "def", text: p[1] }]))
    open = []; moves = 0; found = 0; started = 0; done = false; lock = false
    clearInterval(timer)
    grid.replaceChildren(...cards.map((c, i) => {
      const b = document.createElement("button")
      b.type = "button"
      b.className = "card " + c.kind
      const back = document.createElement("span"); back.className = "back"; back.textContent = "?"
      const front = document.createElement("span"); front.className = "front"; front.textContent = c.text
      b.append(back, front)
      b.onclick = () => flip(i)
      return b
    }))
    status()
  }

  function flip(i) {
    const c = cards[i], el = grid.children[i]
    if (lock || done || c.matched || open.includes(i)) return
    if (!started) { started = Date.now(); timer = setInterval(status, 500) }
    open.push(i)
    el.classList.add("open")
    if (open.length < 2) return
    moves++
    const [a, b] = open.map((j) => cards[j])
    if (a.pair === b.pair) {
      for (const j of open) { cards[j].matched = true; grid.children[j].classList.add("matched") }
      open = []
      found++
      if (found === cards.length / 2) {
        done = Date.now()
        clearInterval(timer)
        if (send) send("won", { moves, seconds: seconds() })
      }
    } else {
      lock = true
      setTimeout(() => { for (const j of open) grid.children[j].classList.remove("open"); open = []; lock = false }, 1100)
    }
    status()
  }

  newBtn.onclick = newGame

  return {
    update(data, setTriggerValue) {
      text = data?.text ?? {}
      send = setTriggerValue
      const key = JSON.stringify(data?.pairs ?? [])
      if (key !== poolKey) { pool = data?.pairs ?? []; poolKey = key; newGame() } else status()
    },
    destroy() { clearInterval(timer) },
  }
}

export default function (component) {
  const { data, parentElement, setTriggerValue } = component
  let game = games.get(parentElement)
  if (!game) { game = makeMatch(parentElement); games.set(parentElement, game) }
  game.update(data, setTriggerValue)
  return () => { game.destroy(); games.delete(parentElement) }
}
"""

# Term Match pairs: course terms (kept in English, as on the slides) and their meaning.
PAIRS = [
    ("Hallucination", {"en": "An unsupported generated claim",
                       "ru": "Неподтверждённое сгенерированное утверждение",
                       "ko": "근거 없이 생성된 주장"}),
    ("Temperature", {"en": "Changes how random sampling is, not how true it is",
                     "ru": "Меняет случайность выбора, а не истинность",
                     "ko": "샘플링의 무작위성을 바꿀 뿐, 정확성은 바꾸지 않음"}),
    ("Recall@k", {"en": "Relevant items in the top k ÷ all known relevant items",
                  "ru": "Релевантные в топ-k ÷ все известные релевантные",
                  "ko": "상위 k개 중 관련 항목 수 ÷ 전체 관련 항목 수"}),
    ("Reciprocal rank", {"en": "1 ÷ the rank of the first relevant result",
                         "ru": "1 ÷ позиция первого релевантного результата",
                         "ko": "1 ÷ 첫 번째 관련 결과의 순위"}),
    ("Chunk overlap", {"en": "Repeats boundary text so meaning isn't cut at a split",
                       "ru": "Повтор текста на границах, чтобы смысл не разрезался",
                       "ko": "경계 텍스트를 반복해 분할 지점에서 의미가 끊기지 않게 함"}),
    ("Cosine similarity", {"en": "A ranking signal, not a verdict of correctness",
                           "ru": "Сигнал для ранжирования, а не доказательство верности",
                           "ko": "순위 신호일 뿐, 정답 판정이 아님"}),
    ("Zero-shot prompt", {"en": "One explicit instruction, no examples",
                          "ru": "Одна чёткая инструкция без примеров",
                          "ko": "예시 없이 명확한 지시 하나"}),
    ("Few-shot prompt", {"en": "An instruction plus a few worked examples",
                         "ru": "Инструкция плюс несколько примеров",
                         "ko": "지시와 몇 개의 예시"}),
    ("Agent loop", {"en": "The model picks an action, calls a tool, observes, repeats",
                    "ru": "Модель выбирает действие, вызывает инструмент, наблюдает, повторяет",
                    "ko": "모델이 행동을 고르고, 도구를 호출하고, 결과를 보고 반복함"}),
    ("Fixed workflow", {"en": "Code decides the path in advance",
                        "ru": "Путь заранее задаёт код",
                        "ko": "코드가 경로를 미리 정함"}),
    ("Tokenizer", {"en": "Splits text into tokens; each model has its own",
                   "ru": "Делит текст на токены; у каждой модели свой",
                   "ko": "텍스트를 토큰으로 나눔; 모델마다 다름"}),
    ("Abstention", {"en": "Saying the evidence is insufficient instead of guessing",
                    "ru": "Сказать, что данных недостаточно, а не угадывать",
                    "ko": "추측하지 않고 근거가 부족하다고 말함"}),
    ("RAG", {"en": "Retrieve evidence, then answer from it",
             "ru": "Найти доказательства и отвечать на их основе",
             "ko": "근거를 검색한 뒤 그것에 근거해 답함"}),
    ("Embedding", {"en": "A vector for meaning: similar texts end up close together",
                   "ru": "Вектор смысла: похожие тексты оказываются рядом",
                   "ko": "의미를 담은 벡터: 비슷한 글은 서로 가까움"}),
]

_RUNNER = st.components.v2.component("lecture_runner", html=RUNNER_HTML, css=RUNNER_CSS,
                                     js=RUNNER_JS)
_MATCH = st.components.v2.component("lecture_match", html=MATCH_HTML, css=MATCH_CSS,
                                    js=MATCH_JS)
GAMES = ["runner", "match"]


@st.cache_resource
def site_records() -> dict:
    """Best results on this server since it started (shared by all visitors)."""
    return {"lock": threading.Lock(), "runner": 0, "match": None}


def _runner_over():
    result = st.session_state["game_runner"].over or {}
    score = int(result.get("score", 0))
    st.session_state.best_runner = max(st.session_state.get("best_runner", 0), score)
    records = site_records()
    with records["lock"]:
        records["runner"] = max(records["runner"], score)


def _match_won():
    result = st.session_state["game_match"].won or {}
    moves = int(result.get("moves", 0))
    if moves:
        mine = st.session_state.get("best_match")
        st.session_state.best_match = moves if mine is None else min(mine, moves)
        records = site_records()
        with records["lock"]:
            records["match"] = moves if records["match"] is None else min(records["match"], moves)
        st.session_state.match_just_won = moves  # shown in the dialog (callbacks can't draw)


def _games():
    st.caption(t("games_caption"))
    game = st.segmented_control(t("games_pick"), GAMES, format_func=lambda g: t(f"game_{g}"),
                                key="game_pick", required=True, label_visibility="collapsed")
    records = site_records()
    if game == "match":
        st.caption(t("match_help"))
        _MATCH(key="game_match", width="stretch", on_won_change=_match_won, data={
            "pairs": [[term, meaning[i18n.lang()]] for term, meaning in PAIRS],
            "text": {"moves": t("match_moves"), "time": t("match_time"),
                     "won": t("match_won"), "new": t("match_new")}})
        won = st.session_state.pop("match_just_won", None)
        if won:
            st.success(t("match_won_toast", moves=won), icon=":material/celebration:")
        mine, site = st.session_state.get("best_match"), records["match"]
        st.caption(t("best_match", mine=mine if mine is not None else "–",
                     site=site if site is not None else "–"))
    else:
        st.caption(t("runner_help"))
        _RUNNER(key="game_runner", width="stretch", on_over_change=_runner_over, data={
            "best": st.session_state.get("best_runner", 0),
            "text": {"start": t("runner_start"), "over": t("runner_over"),
                     "restart": t("runner_restart"), "score": t("runner_score"),
                     "best": t("runner_best")}})
        st.caption(t("best_runner", mine=st.session_state.get("best_runner", 0),
                     site=records["runner"]))


# A dialog's title is fixed when it is defined: one per interface language.
DIALOGS = {code: st.dialog(title, width="large", icon=":material/sports_esports:")(_games)
           for code, title in (("en", "Mini games"), ("ru", "Мини-игры"), ("ko", "미니 게임"))}


def open_games():
    st.session_state.setdefault("game_pick", "runner")
    DIALOGS[i18n.lang()]()
