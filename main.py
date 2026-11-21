"""GTA VI trophy rarity forecast: earners-over-tracked decomposition.

Model, per trophy, in plain words:
  rate(t) = earners(t) / players_tracked(t).
  Both earners and tracked players only grow, but at different speeds:
  tracked players jump when Exophase refreshes its page (launch night it sat
  at 638 players for hours, then jumped to 3,868, and every rate but the one
  at 100% fell), while earners grow as players actually play. So instead of
  extrapolating the rate itself, this program fits, per trophy, how earners
  grow with log time since launch, fits how tracked players grow the same
  way, and divides the two forecasts. It also fits the rate's own log-time
  trend and averages the two in logit space, shrunk toward the last observed
  rate when the fit is noisy. Rare trophies borrow strength from their tier:
  a trophy's slope is shrunk toward the pooled slope of its tier.

When the whole history is flat (no read ever moved, as on launch night
before Exophase's first refresh), there is no trend to fit, but a refresh
is overdue: the median is diluted toward the rate the refresh would leave
(earners roughly unchanged, tracked players up several-fold), and the
spread is widened a lot, mostly downward. A trophy at 100% stays at 100%:
everyone earns the opening trophy.

Spread: a random-walk term (0.5 logit per sqrt day, as in the starter),
plus the fit's residual spread, plus a term for refresh jumps seen in the
history. Quantiles are the median on the logit scale shifted by the normal
quantiles, rounded to 4 decimals half up, as the starter rounds them.

Runs offline, standard library only, deterministic.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]
LAUNCH = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)


def clamp(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg):
    return 100.0 / (1.0 + math.exp(-lg))


def round4(x):
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def hours_since_launch(t):
    return max(0.0, (t - LAUNCH).total_seconds() / 3600.0)


def wls(xs, ys, ws):
    """Weighted least squares y ~ a + b*x. Returns (a, b, resid_std)."""
    sw = sum(ws)
    if sw <= 0 or len(xs) < 2:
        return None
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-12:
        return (my, 0.0, 0.0)
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    b = sxy / sxx
    a = my - b * mx
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    var = sum(w * r * r for w, r in zip(ws, resid)) / sw
    return (a, b, math.sqrt(max(var, 0.0)))


def forecast_case(case, name):
    trophies = case["trophies"]
    history = sorted(case["history"], key=lambda r: r["slot"])
    horizon_days = case["horizon_minutes"] / 1440.0
    target = parse_time(case["target"])
    h_target = hours_since_launch(target)
    x_target = math.log1p(h_target)

    xs = []
    ns = []
    last_time = None
    for read in history:
        t = parse_time(read["slot"])
        xs.append(math.log1p(hours_since_launch(t)))
        ns.append(read["players_tracked"])
        last_time = t
    last = history[-1]
    last_n = last["players_tracked"]

    ws = []
    for read in history:
        t = parse_time(read["slot"])
        age_h = max(0.0, (last_time - t).total_seconds() / 3600.0)
        ws.append(0.5 ** (age_h / 12.0))

    # Is the history flat? (Every rate and the tracked count never moved.)
    flat = True
    for read in history:
        if read["players_tracked"] != last_n:
            flat = False
            break
        if read["rates"] != last["rates"]:
            flat = False
            break

    # Refresh jumps in the history: tracked players jumping between reads.
    jump = 0.0
    for a, b in zip(ns, ns[1:]):
        if a and b:
            jump = max(jump, abs(math.log(b / a)))

    # Fit log tracked players vs log time (shared across trophies).
    n_fit = None
    logn = [math.log(max(n, 1)) for n in ns]
    if len(xs) >= 2 and len(set(xs)) > 1:
        n_fit = wls(xs, logn, ws)

    # First pass: per-trophy fits, to pool slopes by tier.
    tier_slopes = {}
    per_trophy = []
    for i, trophy in enumerate(trophies):
        series = []
        for j, read in enumerate(history):
            r = read["rates"][i]
            if r is None or read["players_tracked"] <= 0:
                continue
            series.append((xs[j], ws[j], r, read["players_tracked"]))
        per_trophy.append(series)
        if len(series) >= 3:
            ys = [math.log(max(r / 100.0 * n, 0.5)) for _, _, r, n in series]
            fit = wls([s[0] for s in series], ys, [s[1] for s in series])
            if fit:
                tier_slopes.setdefault(trophy["tier"], []).append(fit[1])
    tier_slope = {t: sum(v) / len(v) for t, v in tier_slopes.items()}

    rows = []
    for i, trophy in enumerate(trophies):
        series = per_trophy[i]
        last_rate = last["rates"][i]
        if last_rate is None:
            last_rate = 0.0
        # A trophy everyone holds stays held.
        if last_rate >= 99.99:
            rows.append({"id": trophy["id"], "q": [round4(99.9), round4(99.97), 100, 100, 100]})
            continue

        if flat:
            # No trend to fit: a refresh is overdue. Earners roughly
            # unchanged, tracked players up several-fold, so the rate
            # falls hard; keep a wide, downward-heavy spread.
            median = clamp(last_rate * 0.22)
            sigma = 1.6 + 0.5 * math.sqrt(horizon_days)
            lg = logit(median)
            rows.append({"id": trophy["id"], "q": [round4(inv_logit(lg + z * sigma)) for z in Z]})
            continue

        # Trend fit on logit(rate) vs log time.
        trend = None
        resid = 0.3
        if len(series) >= 3 and len(set(s[0] for s in series)) > 1:
            ys = [logit(s[2]) for s in series]
            trend = wls([s[0] for s in series], ys, [s[1] for s in series])
            if trend:
                resid = trend[2]

        # Earners fit, slope shrunk toward the tier's pooled slope.
        en_rate = None
        if len(series) >= 3 and n_fit:
            ys = [math.log(max(s[2] / 100.0 * s[3], 0.5)) for s in series]
            fit = wls([s[0] for s in series], ys, [s[1] for s in series])
            if fit:
                a_e, b_e, _ = fit
                b_group = tier_slope.get(trophy["tier"])
                if b_group is not None:
                    b_e = (len(series) * b_e + 3.0 * b_group) / (len(series) + 3.0)
                log_e_pred = a_e + b_e * x_target
                log_n_pred = n_fit[0] + n_fit[1] * x_target
                en_rate = clamp(100.0 * math.exp(log_e_pred - log_n_pred))

        # Combine the two forecasts in logit space, shrunk to the last rate.
        candidates = []
        if trend:
            candidates.append(trend[0] + trend[1] * x_target)
        if en_rate is not None:
            candidates.append(logit(en_rate))
        if not candidates:
            median = clamp(last_rate)
            sigma = 0.5 * math.sqrt(horizon_days) + 0.5
        else:
            lg = sum(candidates) / len(candidates)
            # Shrink toward persistence when the fit is noisy.
            shrink = 1.0 / (1.0 + resid)
            lg = shrink * lg + (1.0 - shrink) * logit(last_rate)
            median = clamp(inv_logit(lg))
            # Never let the median run away from the last read.
            lo = clamp(last_rate * 0.05)
            hi = clamp(last_rate * 20.0) if last_rate > 0 else 1.0
            median = min(hi, max(lo, median))
            sigma = math.sqrt((0.5 * math.sqrt(horizon_days)) ** 2 + min(resid, 1.5) ** 2 + (0.5 if jump > 0.2 else 0.15) ** 2)
        lg = logit(median)
        qs = [round4(inv_logit(lg + z * sigma)) for z in Z]
        qs = sorted(max(0, min(100, q)) for q in qs)
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
