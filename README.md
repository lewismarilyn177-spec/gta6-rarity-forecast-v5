# GTA6 Trophy Rarity Forecast

This entry forecasts trophy rarity percentiles for Grand Theft Auto VI on PS5 using Exophase trophy tracker data.

## What it does

Reads `.in` files from `$MAKEFUN_INPUT_DIR` (default `/input`), each containing a forecast case with:
- Game metadata (release date, platform)
- 55 trophy definitions (id, name, tier, etc.)
- Historical reads from Exophase: timestamp, players tracked, and per-trophy rarity rates (2 decimals)
- A target timestamp to forecast

Writes one `.out` file per case to `$MAKEFUN_OUTPUT_DIR` (default `/output`) with five quantiles (0.05, 0.25, 0.5, 0.75, 0.95) for each trophy's rarity at the target time.

## Earner Reconstruction (Key Insight)

Exophase displays each trophy's rarity as a percentage rounded to **2 decimals** (e.g., `1.25%`), alongside the **players-tracked count** `P`. The true earner count `e` is an integer satisfying:

```
round2(e / P * 100) = displayed_rate
```

Since `e` is integer and `round2` is deterministic, each displayed rate maps to a **unique** `e` (when one exists). We recover `e` by scanning integers near `rate/100 * P` (from `floor-2` to `ceil+2`) and picking the one whose `round2(e/P*100)` matches exactly. Fallback: `Math.round(rate/100*P)`.

This works because the Exophase page is **frozen between refreshes**: when the player count jumps (e.g., 638 → 3868), the earners `e` stay fixed—only the denominator changes, diluting every rate by `638/3868`.

## Two Branches

### Branch A — Stale Launch Page (practice cases p00, p01)

**Condition**: Every historical read has identical `players_tracked` and identical rates as the last read, **and** the last `players_tracked < 2000` (the launch-night page at 638 tracked).

**Why point mass?** The case straddles the documented first refresh to 3,868 players. Earners are known exactly (reconstructed from the frozen page). The refresh is a deterministic re-render: `e` stays the same, `P` becomes 3868, so the new rate `r = round2(e/3868*100)` is known with zero uncertainty.

**Forecast**: If last rate ≥ 99.995 → all five quantiles = 100. Otherwise all five quantiles = `r` (a point mass).

### Branch B — General Case (final evaluation case)

**Logit-linear projection** with pooled shrinkage:

1. **Collapse history** into distinct refresh states (new state when `players_tracked` or any rate changes).
2. **Per-trophy fit**: `u = ln(max(hours_since_release, 0.5))`, `y = logit(rate clamped to [0.01, 99.99])`. OLS slope `b_i` with ≥3 distinct points.
3. **Pooled slope**: Median of finite `b_i`, clamped to `[-1.0, 1.5]`.
4. **Shrinkage**: `b = w*b_i + (1-w)*b_pool` with `w = min(0.7, points/8)`; if `b_i` undefined, `b = b_pool`.
5. **Project center**: `du = ln(x_T / x_last)`, `move = b * du * damp`, `damp = 1/(1+0.12*du)` clamped to `[-1.8, 1.8]`, `y_T = y_last + move`.
6. **Earner floor**: Earners never decrease. Project `P_T` by fitting `log(P)` vs `ln(x)` over last two states (damped growth, clamped to `[P_last, 6*P_last]`). Floor rate = `100 * e_last / P_T`; raise `y_T` if floor is higher.
7. **Spread**: `sigma = 0.30 + 0.35*ln(1+du/2) + res`, where `res = min(0.35, 1.5*RMS_residual)` if fit has ≥3 points else `0.15`; clamp `sigma` to `[0.30, 1.10]`.
8. **Quantiles**: `inv_logit(y_T + z*sigma)` for `z ∈ {-1.6449, -0.6745, 0, 0.6745, 1.6449}`. Round to 4 decimals, enforce non-decreasing.

## Determinism

- No randomness, no `Date.now()`, no network, no child processes.
- Pure TypeScript (stripped at runtime via `node --experimental-strip-types`).
- Standard library only, no dependencies.
- Output is byte-for-byte deterministic for a given input.