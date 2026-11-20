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
