const games = new WeakMap()

// Speeds in pixels per second. Starts calm and speeds up slowly (players found it too fast).
const START_SPEED = 230, MAX_SPEED = 540, SPEED_UP = 5, GRAVITY = 2000, JUMP = 720

function makeRunner(root) {
  const canvas = root.querySelector("canvas")
  const ctx = canvas.getContext("2d")
  const host = root.host ?? root
  const H = 230, GROUND = H - 34
  let W = 640, text = {}, best = 0, send = null
  let state = "ready", raf = 0, last = performance.now(), clock = 0
  let p, obstacles, stars, speed, distance, bonus, nextSpawn, nextStar, pops

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
    obstacles = []; stars = []; pops = []
    speed = START_SPEED; distance = 0; bonus = 0; nextSpawn = 520; nextStar = 760
  }

  const score = () => Math.floor(distance / 10) + bonus

  function jump() {
    if (state !== "running") { reset(); state = "running"; return }
    if (p.onGround) { p.vy = -JUMP; p.onGround = false }
  }

  function spawn() {
    const roll = Math.random()
    if (roll < 0.4 || score() < 80) {         // a software bug crawling on the track
      obstacles.push({ kind: "bug", x: W + 20, y: GROUND - 28, w: 46, h: 28 })
    } else if (roll < 0.75) {                 // a traffic cone: jump over it
      const h = 36 + Math.random() * 14
      obstacles.push({ kind: "cone", x: W + 20, y: GROUND - h, w: 32, h })
    } else {                                  // a hallucination ghost at head height: run under it
      obstacles.push({ kind: "ghost", x: W + 20, y: GROUND - 94 - Math.random() * 10, w: 40, h: 36 })
    }
  }

  function hit(a, b, pad) {
    return a.x + pad < b.x + b.w && a.x + a.w - pad > b.x && a.y + pad < b.y + b.h && a.y + a.h - pad > b.y
  }

  function update(dt) {
    speed = Math.min(MAX_SPEED, speed + SPEED_UP * dt)
    distance += speed * dt
    p.vy += GRAVITY * dt
    p.y += p.vy * dt
    if (p.y >= GROUND - p.h) { p.y = GROUND - p.h; p.vy = 0; p.onGround = true }

    nextSpawn -= speed * dt
    if (nextSpawn <= 0) { spawn(); nextSpawn = speed * (1.05 + Math.random() * 0.95) + 170 }
    nextStar -= speed * dt
    if (nextStar <= 0) {
      const ghostNear = obstacles.some(o => o.kind === "ghost" && o.x > W - 160)
      if (!ghostNear) stars.push({ x: W + 20, y: GROUND - 124, w: 30, h: 30 })
      nextStar = speed * (2.0 + Math.random() * 2.2)
    }
    for (const o of obstacles) o.x -= speed * dt
    for (const s of stars) s.x -= speed * dt
    for (const q of pops) { q.t -= dt; q.y -= 40 * dt }
    obstacles = obstacles.filter(o => o.x + o.w > -10)
    stars = stars.filter(s => s.x + s.w > -10 && !s.taken)
    pops = pops.filter(q => q.t > 0)

    for (const s of stars) {
      if (!s.taken && hit(p, s, 0)) { s.taken = true; bonus += 25; pops.push({ x: s.x, y: s.y, t: 0.8 }) }
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

  // ---- characters ----
  function drawBug(x, y, w, h, c) {
    const walk = Math.sin(clock * 18) * 2
    ctx.strokeStyle = c.dark; ctx.lineWidth = 2
    for (let i = 0; i < 3; i++) {                       // six legs
      const lx = x + w * 0.3 + i * w * 0.2, sway = i % 2 ? walk : -walk
      ctx.beginPath(); ctx.moveTo(lx, y + h - 8); ctx.lineTo(lx - 4 + sway, y + h); ctx.stroke()
      ctx.beginPath(); ctx.moveTo(lx + 2, y + h - 8); ctx.lineTo(lx + 6 - sway, y + h); ctx.stroke()
    }
    ctx.fillStyle = c.red                               // shell
    ctx.beginPath(); ctx.ellipse(x + w / 2 + 3, y + h / 2 + 2, w / 2 - 3, h / 2 - 2, 0, 0, Math.PI * 2); ctx.fill()
    ctx.strokeStyle = c.dark; ctx.lineWidth = 1.5
    ctx.beginPath(); ctx.moveTo(x + w / 2 + 3, y + 5); ctx.lineTo(x + w / 2 + 3, y + h - 3); ctx.stroke()
    ctx.fillStyle = c.dark                              // spots
    ctx.beginPath(); ctx.arc(x + w / 2 - 3, y + h / 2, 2.5, 0, 7); ctx.arc(x + w / 2 + 10, y + h / 2 + 3, 2.5, 0, 7); ctx.fill()
    ctx.beginPath(); ctx.arc(x + 7, y + h / 2 + 3, h * 0.3, 0, 7); ctx.fill()   // head (facing the robot)
    ctx.strokeStyle = c.dark; ctx.lineWidth = 1.5     // antennae
    ctx.beginPath(); ctx.moveTo(x + 4, y + h / 2 - 3); ctx.lineTo(x - 2, y + 1); ctx.stroke()
    ctx.beginPath(); ctx.moveTo(x + 8, y + h / 2 - 3); ctx.lineTo(x + 6, y - 1); ctx.stroke()
    ctx.fillStyle = c.white
    ctx.beginPath(); ctx.arc(x + 4, y + h / 2 + 2, 2, 0, 7); ctx.fill()
  }

  function drawCone(x, y, w, h, c) {
    ctx.fillStyle = c.orange
    ctx.beginPath(); ctx.moveTo(x + w / 2, y); ctx.lineTo(x + w - 3, y + h - 5); ctx.lineTo(x + 3, y + h - 5); ctx.closePath(); ctx.fill()
    ctx.fillStyle = c.white                            // reflective stripes
    for (const f of [0.38, 0.66]) {
      const yy = y + h * f, half = (w / 2 - 3) * f
      ctx.fillRect(x + w / 2 - half, yy, half * 2, 4)
    }
    ctx.fillStyle = c.orange; rounded(x, y + h - 6, w, 6, 2); ctx.fill()   // base
  }

  function drawGhost(x, y, w, h, c) {
    const bob = Math.sin(clock * 4) * 3
    y += bob
    ctx.globalAlpha = 0.9; ctx.fillStyle = c.violet
    ctx.beginPath()
    ctx.arc(x + w / 2, y + w / 2, w / 2, Math.PI, 0)
    ctx.lineTo(x + w, y + h)
    const waves = 4, step = w / waves
    for (let i = 0; i < waves; i++) {
      const x1 = x + w - step * i, x2 = x1 - step
      ctx.quadraticCurveTo(x1 - step / 2, y + h - (i % 2 ? -4 : 7), x2, y + h)
    }
    ctx.closePath(); ctx.fill(); ctx.globalAlpha = 1
    ctx.fillStyle = c.white
    ctx.beginPath(); ctx.ellipse(x + 12, y + 14, 4, 5, 0, 0, 7); ctx.ellipse(x + 24, y + 14, 4, 5, 0, 0, 7); ctx.fill()
    ctx.fillStyle = c.dark
    ctx.beginPath(); ctx.arc(x + 11, y + 15, 2, 0, 7); ctx.arc(x + 23, y + 15, 2, 0, 7); ctx.fill()
    ctx.beginPath(); ctx.ellipse(x + 18, y + 24, 3, 2.5, 0, 0, 7); ctx.fill()  // "oo" mouth
  }

  function drawStar(cx, cy, r, c, spin) {
    ctx.save(); ctx.translate(cx, cy); ctx.rotate(spin)
    ctx.globalAlpha = 0.25; ctx.fillStyle = c.gold
    ctx.beginPath(); ctx.arc(0, 0, r + 6 + Math.sin(clock * 6) * 2, 0, 7); ctx.fill()   // glow
    ctx.globalAlpha = 1; ctx.fillStyle = c.gold
    ctx.beginPath()
    for (let i = 0; i < 10; i++) {
      const a = -Math.PI / 2 + (i * Math.PI) / 5, rr = i % 2 ? r * 0.45 : r
      ctx.lineTo(Math.cos(a) * rr, Math.sin(a) * rr)
    }
    ctx.closePath(); ctx.fill()
    ctx.strokeStyle = c.goldEdge; ctx.lineWidth = 1.5; ctx.stroke()
    ctx.fillStyle = c.white; ctx.globalAlpha = 0.8
    ctx.beginPath(); ctx.arc(-r * 0.2, -r * 0.25, r * 0.18, 0, 7); ctx.fill()   // shine
    ctx.restore(); ctx.globalAlpha = 1
  }

  function drawRobot(r, c) {
    ctx.fillStyle = c.primary
    ctx.fillRect(r.x + 14, r.y - 8, 2, 8)
    ctx.beginPath(); ctx.arc(r.x + 15, r.y - 9, 3, 0, 7); ctx.fill()
    rounded(r.x, r.y, r.w, r.h - 8, 7); ctx.fill()
    const step = r.onGround && state === "running" ? Math.floor((distance || 0) / 40) % 2 : 0
    ctx.fillRect(r.x + 5, r.y + r.h - 9, 7, 9 - step * 4)
    ctx.fillRect(r.x + 18, r.y + r.h - 9, 7, 5 + step * 4)
    ctx.fillStyle = c.white; rounded(r.x + 5, r.y + 6, 20, 10, 4); ctx.fill()
    ctx.fillStyle = "#000000"  // black eyes on the white visor, in light and dark mode
    ctx.fillRect(r.x + 10, r.y + 9, 3, 4); ctx.fillRect(r.x + 18, r.y + 9, 3, 4)
  }

  function colors() {
    return {
      text: css("--st-text-color", "#1e1b4b"), white: "#ffffff", dark: "#1f1d2e",
      panel: css("--st-secondary-background-color", "#f5f3ff"),
      primary: css("--st-primary-color", "#6d28d9"), border: css("--st-border-color", "#d4d4d8"),
      red: css("--st-red-color", "#dc2626"), orange: css("--st-orange-color", "#f97316"),
      violet: css("--st-violet-color", "#8b5cf6"), gold: "#fbbf24", goldEdge: "#b45309",
      green: css("--st-green-color", "#16a34a"), font: css("--st-font", "sans-serif"),
    }
  }

  function draw() {
    const c = colors(), font = c.font
    ctx.clearRect(0, 0, W, H)
    ctx.fillStyle = c.panel; rounded(0, 0, W, H, 14); ctx.fill()

    ctx.fillStyle = c.border                            // track with moving dashes
    ctx.fillRect(0, GROUND, W, 2)
    const off = (distance || 0) % 46
    for (let x = -off; x < W; x += 46) ctx.fillRect(x, GROUND + 9, 18, 2)

    for (const s of stars) drawStar(s.x + s.w / 2, s.y + s.h / 2, 14, c, clock * 2)
    for (const o of obstacles) {
      if (o.kind === "bug") drawBug(o.x, o.y, o.w, o.h, c)
      else if (o.kind === "cone") drawCone(o.x, o.y, o.w, o.h, c)
      else drawGhost(o.x, o.y, o.w, o.h, c)
    }
    drawRobot(p, c)

    ctx.textAlign = "center"; ctx.textBaseline = "middle"
    ctx.font = `700 15px ${font}`; ctx.fillStyle = c.goldEdge
    for (const q of pops) { ctx.globalAlpha = Math.max(0, q.t / 0.8); ctx.fillText("+25", q.x + 13, q.y); ctx.globalAlpha = 1 }

    ctx.fillStyle = c.text; ctx.textAlign = "right"; ctx.font = `600 14px ${font}`
    ctx.fillText(`${text.score ?? "Score"} ${score()}   ${text.best ?? "Best"} ${best}`, W - 14, 20)

    if (state === "running") return
    const boxW = Math.min(480, W - 20), boxH = state === "ready" ? 104 : 72
    ctx.globalAlpha = 0.92; ctx.fillStyle = c.panel     // keeps the text readable over the track
    rounded(W / 2 - boxW / 2, 40, boxW, boxH, 12); ctx.fill(); ctx.globalAlpha = 1
    ctx.textAlign = "center"; ctx.fillStyle = c.text
    if (state === "over") {
      ctx.font = `700 22px ${font}`; ctx.fillText(`${text.over ?? "Game over"}: ${score()}`, W / 2, 64)
      ctx.font = `500 14px ${font}`; ctx.fillText(text.restart ?? "", W / 2, 94)
      return
    }
    ctx.font = `600 15px ${font}`; ctx.fillText(text.start ?? "", W / 2, 60)
    // legend: what each character means
    const items = [["jump", text.jump ?? "Jump over"], ["under", text.under ?? "Run under"],
                   ["star", text.bonus ?? "+25"]]
    const slot = boxW / 3, top = 84
    ctx.font = `500 13px ${font}`; ctx.textAlign = "left"
    items.forEach(([kind, label], i) => {
      const x0 = W / 2 - boxW / 2 + slot * i + 14
      if (kind === "jump") { drawBug(x0, top + 4, 30, 18, c); drawCone(x0 + 34, top - 4, 20, 26, c) }
      else if (kind === "under") drawGhost(x0 + 8, top - 6, 26, 24, c)
      else drawStar(x0 + 18, top + 9, 10, c, 0)
      ctx.fillStyle = c.text
      ctx.fillText(label, x0 + (kind === "jump" ? 60 : 40), top + 11)
    })
  }

  function frame(now) {
    const dt = Math.min(0.05, (now - last) / 1000)
    last = now
    clock += dt
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
