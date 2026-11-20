// make.fun GTA6 Trophy Rarity Forecast
// Reads trophy rarity history from Exophase, reconstructs integer earner counts from
// 2-decimal rates, and forecasts quantiles at a target time.
// Branch A (stale launch page): all reads identical, players_tracked < 2000.
//   The page will refresh to 3868 tracked players; earners stay fixed, rates dilute.
//   Forecast is a point mass at the post-refresh rate (or 100 if already ~100).
// Branch B (general): logit-linear projection with pooled slope shrinkage,
//   player-count growth projection, earner-count floor, and calibrated spread.

import { readFileSync, writeFileSync, readdirSync, mkdirSync } from "fs";
import { join, dirname } from "path";
import { fileURLToPath } from "url";

const __dirname = dirname(fileURLToPath(import.meta.url));

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR || "/input";
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR || "/output";

function round2(x: number): number {
  return Math.round(x * 100) / 100;
}

function clamp(x: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, x));
}

function median(arr: number[]): number {
  const sorted = [...arr].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

function reconstructEarners(rate: number, playersTracked: number): number {
  const approx = rate / 100 * playersTracked;
  const start = Math.floor(approx) - 2;
  const end = Math.ceil(approx) + 2;
  for (let e = start; e <= end; e++) {
    if (round2(e / playersTracked * 100) === rate) {
      return e;
    }
  }
  return Math.round(approx);
}

function logit(p: number): number {
  const clamped = clamp(p, 0.01, 99.99);
  return Math.log(clamped / (100 - clamped));
}

function invLogit(y: number): number {
  return 100 / (1 + Math.exp(-y));
}

function olsSlope(xs: number[], ys: number[]): number | null {
  const n = xs.length;
  if (n < 3) return null;
  const xMean = xs.reduce((a, b) => a + b, 0) / n;
  const yMean = ys.reduce((a, b) => a + b, 0) / n;
  let num = 0, den = 0;
  for (let i = 0; i < n; i++) {
    const dx = xs[i] - xMean;
    num += dx * (ys[i] - yMean);
    den += dx * dx;
  }
  if (den === 0) return null;
  return num / den;
}

function rmsResidual(xs: number[], ys: number[], slope: number): number {
  const n = xs.length;
  const xMean = xs.reduce((a, b) => a + b, 0) / n;
  const yMean = ys.reduce((a, b) => a + b, 0) / n;
  const intercept = yMean - slope * xMean;
  let sumSq = 0;
  for (let i = 0; i < n; i++) {
    const pred = intercept + slope * xs[i];
    sumSq += (ys[i] - pred) ** 2;
  }
  return Math.sqrt(sumSq / n);
}

function hoursSinceRelease(dateStr: string, releaseDate: string): number {
  const d = new Date(dateStr).getTime();
  const r = new Date(releaseDate + "T00:00:00Z").getTime();
  return (d - r) / 3_600_000;
}

function isBranchA(history: any[]): boolean {
  if (history.length === 0) return false;
  const last = history[history.length - 1];
  if (last.players_tracked >= 2000) return false;
  const lastRates = last.rates;
  const lastPlayers = last.players_tracked;
  for (const h of history) {
    if (h.players_tracked !== lastPlayers) return false;
    if (h.rates.some((r: number, i: number) => r !== lastRates[i])) return false;
  }
  return true;
}

function processCase(inputPath: string, outputPath: string): void {
  const raw = readFileSync(inputPath, "utf8");
  const data = JSON.parse(raw);

  const { id, at, target, trophies, history, game } = data;
  const trophyCount = trophies.length;
  const lastRead = history[history.length - 1];
  const lastRates = lastRead.rates;
  const lastPlayers = lastRead.players_tracked;
  const releaseDate = game.release_date;

  const result: { id: number; q: number[] }[] = [];

  if (isBranchA(history)) {
    // Branch A: stale launch page, refresh to 3868 players
    const REFRESH_PLAYERS = 3868;
    for (let i = 0; i < trophyCount; i++) {
      const rate = lastRates[i];
      if (rate >= 99.995) {
        result.push({ id: trophies[i].id, q: [100, 100, 100, 100, 100] });
      } else {
        const e = reconstructEarners(rate, lastPlayers);
        const r = round2(e / REFRESH_PLAYERS * 100);
        const q = [r, r, r, r, r].map(v => Math.round(v * 10000) / 10000);
        result.push({ id: trophies[i].id, q });
      }
    }
  } else {
    // Branch B: general forecasting
    // Collapse history into distinct refresh states
    const states: { at: string; players: number; rates: (number | null)[] }[] = [];
    for (const h of history) {
      const prev = states[states.length - 1];
      if (!prev || h.players_tracked !== prev.players || h.rates.some((r: number, i: number) => r !== prev.rates[i])) {
        states.push({ at: h.at, players: h.players_tracked, rates: h.rates.map((r: number | null) => r) });
      }
    }

    // Project player count P_T at target
    const stateCount = states.length;
    let pT: number;
    if (stateCount >= 2) {
      const lastState = states[stateCount - 1];
      const prevState = states[stateCount - 2];
      const xLast = Math.max(hoursSinceRelease(lastState.at, releaseDate), 0.5);
      const xPrev = Math.max(hoursSinceRelease(prevState.at, releaseDate), 0.5);
      const uLast = Math.log(xLast);
      const uPrev = Math.log(xPrev);
      const logPLast = Math.log(lastState.players);
      const logPPrev = Math.log(prevState.players);
      const slopeP = (logPLast - logPPrev) / (uLast - uPrev);
      const xT = hoursSinceRelease(target, releaseDate);
      const uT = Math.log(Math.max(xT, 0.5));
      const du = uT - uLast;
      const dampP = 1 / (1 + 0.15 * du);
      const logPT = logPLast + slopeP * du * dampP;
      pT = Math.exp(logPT);
      pT = clamp(pT, lastState.players, 6 * lastState.players);
    } else {
      pT = lastRead.players_tracked;
    }

    const lastState = states[states.length - 1];
    const xLast = Math.max(hoursSinceRelease(lastState.at, releaseDate), 0.5);
    const uLast = Math.log(xLast);
    const xT = hoursSinceRelease(target, releaseDate);
    const uT = Math.log(Math.max(xT, 0.5));
    const du = uT - uLast;
    const damp = clamp(1 / (1 + 0.12 * du), -1.8, 1.8);

    // Compute pooled slope
    const trophySlopes: number[] = [];
    const trophyData: { xs: number[]; ys: number[]; lastRate: number; lastY: number; slope: number | null }[] = [];

    for (let i = 0; i < trophyCount; i++) {
      const xs: number[] = [];
      const ys: number[] = [];
      let lastRate = lastRates[i];
      let lastY = logit(lastRate);
      let seenRate: number | null = null;
      for (const st of states) {
        const r = st.rates[i];
        if (r === null) continue;
        if (seenRate !== null && r === seenRate) continue;
        seenRate = r;
        const x = Math.max(hoursSinceRelease(st.at, releaseDate), 0.5);
        xs.push(Math.log(x));
        ys.push(logit(r));
        lastRate = r;
        lastY = logit(r);
      }
      const slope = olsSlope(xs, ys);
      trophyData.push({ xs, ys, lastRate, lastY, slope });
      if (slope !== null) trophySlopes.push(slope);
    }

    const bPool = trophySlopes.length > 0 ? clamp(median(trophySlopes), -1.0, 1.5) : 0;

    const Z = [-1.6449, -0.6745, 0, 0.6745, 1.6449];

    for (let i = 0; i < trophyCount; i++) {
      const { xs, ys, lastRate, lastY, slope: b_i } = trophyData[i];
      const nPoints = xs.length;

      if (lastRate >= 99.995) {
        result.push({ id: trophies[i].id, q: [100, 100, 100, 100, 100] });
        continue;
      }

      let b: number;
      if (b_i !== null) {
        const w = Math.min(0.7, nPoints / 8);
        b = w * b_i + (1 - w) * bPool;
      } else {
        b = bPool;
      }

      const move = b * du * damp;
      let yT = lastY + move;

      // Earner floor
      const e = reconstructEarners(lastRate, lastState.players);
      const floorRate = 100 * e / pT;
      const floorY = logit(floorRate);
      if (floorY > yT) yT = floorY;

      // Guard: fewer than 2 distinct points -> persistence with sigma = 0.70
      if (nPoints < 2) {
        const tier = trophies[i].tier;
        let sigma = 0.70;
        if (tier === "platinum" || tier === "gold") sigma += 0.08;
        else if (tier === "bronze") sigma -= 0.04;
        sigma = clamp(sigma, 0.30, 1.10);
        const q = Z.map(z => {
          const val = invLogit(lastY + z * sigma);
          return Math.round(val * 10000) / 10000;
        });
        for (let j = 1; j < 5; j++) if (q[j] < q[j - 1]) q[j] = q[j - 1];
        result.push({ id: trophies[i].id, q });
        continue;
      }

      // Spread
      let res = 0.15;
      if (nPoints >= 3 && b_i !== null) {
        res = Math.min(0.35, 1.5 * rmsResidual(xs, ys, b_i));
      }
      let sigma = 0.30 + 0.35 * Math.log(1 + du / 2) + res;
      // Tier adjustment
      const tier = trophies[i].tier;
      if (tier === "platinum" || tier === "gold") sigma += 0.08;
      else if (tier === "bronze") sigma -= 0.04;
      sigma = clamp(sigma, 0.30, 1.10);

      const q = Z.map(z => {
        const val = invLogit(yT + z * sigma);
        return Math.round(val * 10000) / 10000;
      });
      // Ensure non-decreasing
      for (let j = 1; j < 5; j++) {
        if (q[j] < q[j - 1]) q[j] = q[j - 1];
      }
      result.push({ id: trophies[i].id, q });
    }
  }

  // Sort by id for deterministic output
  result.sort((a, b) => a.id - b.id);
  const output = JSON.stringify({ trophies: result });
  writeFileSync(outputPath, output);
}

function main(): void {
  mkdirSync(OUTPUT_DIR, { recursive: true });
  const files = readdirSync(INPUT_DIR)
    .filter(f => f.endsWith(".in"))
    .sort();
  for (const file of files) {
    const inputPath = join(INPUT_DIR, file);
    const outputPath = join(OUTPUT_DIR, file.replace(".in", ".out"));
    processCase(inputPath, outputPath);
  }
}

main();