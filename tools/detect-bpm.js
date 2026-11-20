(() => {
  const TARGET_RATE = 22050
  const FRAME = 1024
  const HOP = 256
  const MIN_BPM = 60
  const MAX_BPM = 200
  const WINDOW_SECONDS = 20
  const STEP_SECONDS = 10
  const MIN_SECTION_SECONDS = 30

  const tick = () => new Promise((resolve) => setTimeout(resolve))

  function fft(re, im) {
    const n = re.length

    for (let i = 1, j = 0; i < n; i++) {
      let bit = n >> 1
      for (; j & bit; bit >>= 1) j ^= bit
      j ^= bit
      if (i < j) {
        const tr = re[i]
        re[i] = re[j]
        re[j] = tr
        const ti = im[i]
        im[i] = im[j]
        im[j] = ti
      }
    }

    for (let len = 2; len <= n; len <<= 1) {
      const angle = (-2 * Math.PI) / len
      const wr = Math.cos(angle)
      const wi = Math.sin(angle)
      for (let i = 0; i < n; i += len) {
        let cr = 1
        let ci = 0
        for (let k = 0; k < len / 2; k++) {
          const a = i + k
          const b = a + len / 2
          const xr = re[b] * cr - im[b] * ci
          const xi = re[b] * ci + im[b] * cr
          re[b] = re[a] - xr
          im[b] = im[a] - xi
          re[a] += xr
          im[a] += xi
          const next = cr * wr - ci * wi
          ci = cr * wi + ci * wr
          cr = next
        }
      }
    }
  }

  async function onsetEnvelope(mono, sampleRate, onProgress) {
    const step = Math.max(1, Math.round(sampleRate / TARGET_RATE))
    const rate = sampleRate / step
    const n = Math.floor(mono.length / step)

    const x = new Float32Array(n)
    for (let i = 0; i < n; i++) {
      let sum = 0
      for (let k = 0; k < step; k++) sum += mono[i * step + k]
      x[i] = sum / step
    }

    const hann = Float64Array.from({ length: FRAME }, (_, i) => 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / FRAME))
    const maxBin = Math.floor((5000 * FRAME) / rate)
    const frames = Math.floor((n - FRAME) / HOP)
    const env = new Float64Array(frames)

    const re = new Float64Array(FRAME)
    const im = new Float64Array(FRAME)
    let prev = new Float64Array(maxBin)
    let cur = new Float64Array(maxBin)

    for (let t = 0; t < frames; t++) {
      for (let i = 0; i < FRAME; i++) {
        re[i] = x[t * HOP + i] * hann[i]
        im[i] = 0
      }
      fft(re, im)

      let flux = 0
      for (let b = 1; b < maxBin; b++) {
        cur[b] = Math.log(1 + 1000 * Math.hypot(re[b], im[b]))
        flux += Math.max(0, cur[b] - prev[b])
      }
      env[t] = flux
      ;[prev, cur] = [cur, prev]

      if (t % 4000 === 0) {
        onProgress?.(t / frames)
        await tick()
      }
    }

    const half = Math.round(rate / HOP)
    const sums = new Float64Array(frames + 1)
    for (let t = 0; t < frames; t++) sums[t + 1] = sums[t] + env[t]
    for (let t = 0; t < frames; t++) {
      const from = Math.max(0, t - half)
      const to = Math.min(frames, t + half + 1)
      const mean = (sums[to] - sums[from]) / (to - from)
      env[t] = Math.max(0, env[t] - mean)
    }

    return { env, frameRate: rate / HOP }
  }

  function magnitude(env, cyclesPerFrame) {
    const angle = 2 * Math.PI * cyclesPerFrame
    const cr = Math.cos(angle)
    const ci = Math.sin(angle)
    let pr = 1
    let pi = 0
    let re = 0
    let im = 0

    for (let t = 0; t < env.length; t++) {
      re += env[t] * pr
      im += env[t] * pi
      const next = pr * cr - pi * ci
      pi = pr * ci + pi * cr
      pr = next
    }
    return Math.hypot(re, im)
  }

  function estimate(env, frameRate, coarseStep, refine) {
    const perFrame = (bpm) => bpm / 60 / frameRate

    const score = (bpm) => {
      const prior = Math.exp(-0.5 * Math.log2(bpm / 120) ** 2)
      const sum = magnitude(env, perFrame(bpm)) + 0.5 * magnitude(env, perFrame(bpm * 2)) + 0.33 * magnitude(env, perFrame(bpm * 3))
      return sum * prior
    }

    const search = (from, to, step) => {
      let best = from
      let bestScore = -1
      for (let bpm = from; bpm <= to; bpm += step) {
        const s = score(bpm)
        if (s > bestScore) {
          bestScore = s
          best = bpm
        }
      }
      return best
    }

    const coarse = search(MIN_BPM, MAX_BPM, coarseStep)
    return refine ? search(coarse - coarseStep, coarse + coarseStep, 0.01) : coarse
  }

  const sameTempo = (a, b) => {
    const octaves = Math.log2(a / b)
    return Math.abs(octaves - Math.round(octaves)) < 0.02
  }

  async function detectSections(mono, sampleRate, onProgress) {
    const { env, frameRate } = await onsetEnvelope(mono, sampleRate, (p) => onProgress?.(p * 0.6))

    const size = Math.round(WINDOW_SECONDS * frameRate)
    const step = Math.round(STEP_SECONDS * frameRate)
    const windows = []
    for (let from = 0; from + size <= env.length; from += step) {
      windows.push({ from, bpm: estimate(env.subarray(from, from + size), frameRate, 0.5, false) })
      if (windows.length % 20 === 0) {
        onProgress?.(0.6 + 0.3 * Math.min(1, from / env.length))
        await tick()
      }
    }
    if (windows.length === 0) windows.push({ from: 0, bpm: estimate(env, frameRate, 0.5, false) })

    const groups = []
    for (const w of windows) {
      const last = groups[groups.length - 1]
      if (last && sameTempo(w.bpm, last.bpm)) last.to = w.from + step
      else groups.push({ from: w.from, to: w.from + step, bpm: w.bpm })
    }

    const minFrames = MIN_SECTION_SECONDS * frameRate
    const kept = []
    for (const g of groups) {
      const last = kept[kept.length - 1]
      if (last && (g.to - g.from < minFrames || sameTempo(g.bpm, last.bpm))) last.to = g.to
      else kept.push({ ...g })
    }

    const sections = []
    for (const g of kept) {
      const to = Math.min(env.length, g.to + size - step)
      const bpm = estimate(env.subarray(g.from, to), frameRate, 0.25, true)
      sections.push({ start: g.from / frameRate, end: to / frameRate, bpm })
    }
    for (let i = 0; i < sections.length - 1; i++) sections[i].end = sections[i + 1].start
    onProgress?.(1)
    return sections
  }

  window.detectSections = detectSections
})()
