"""Plateau / expand rules for the CIFAR iso-shape scale-gap study.

Decisions are functions of mean best-test-acc tables so they can be unit-tested
without training.
"""

from __future__ import annotations

from dataclasses import dataclass

DELTA_PP = 0.5
H_CAP = 8192
S_CAP = 1024
CELL_SEC_CAP = 4 * 3600
SCALE_H = (128, 256, 512, 1024, 2048)
S_CROSS = (8, 32, 128, 256)
PRIMARY_S = 32


@dataclass(frozen=True)
class ExpandDecision:
    expand_h: bool
    next_h: int | None
    expand_s: bool
    next_s_values: tuple[int, ...]
    s_h: int | None
    stop: bool
    reasons: tuple[str, ...]

    def summary_lines(self) -> list[str]:
        lines = ["# Decision"]
        if self.expand_h and self.next_h is not None:
            lines.append(f"EXPAND H: next H={self.next_h}")
        if self.expand_s and self.next_s_values:
            h = self.s_h if self.s_h is not None else "?"
            vals = ",".join(str(s) for s in self.next_s_values)
            lines.append(f"EXPAND S: at H={h} run S={vals}")
        if self.stop:
            lines.append("STOP")
        for r in self.reasons:
            lines.append(f"  · {r}")
        return lines


def _pp(delta_acc: float) -> float:
    return delta_acc * 100.0


def decide_expand(
    *,
    float_by_h: dict[int, float],
    tally_by_hs: dict[tuple[int, int], float],
    delta_pp: float = DELTA_PP,
    h_cap: int = H_CAP,
    s_cap: int = S_CAP,
    primary_s: int = PRIMARY_S,
    longest_cell_sec: float | None = None,
    cell_sec_cap: float = CELL_SEC_CAP,
) -> ExpandDecision:
    """Return the next H/S rung, or STOP.

    ``float_by_h``: hidden_dim → mean best test acc
    ``tally_by_hs``: (hidden_dim, S) → mean best test acc
    """
    reasons: list[str] = []
    hs = sorted({h for (h, s) in tally_by_hs if s == primary_s})

    expand_h = False
    next_h: int | None = None
    tally_climbing = False
    gap_closed_last = False
    gap_plateau = False

    if len(hs) < 2:
        reasons.append("need at least two H at S=%d to judge a width plateau" % primary_s)
        if hs:
            cand = hs[-1] * 2
            if cand <= h_cap:
                expand_h = True
                next_h = cand
    else:
        t0, t1 = tally_by_hs[(hs[-2], primary_s)], tally_by_hs[(hs[-1], primary_s)]
        tally_gain = _pp(t1 - t0)
        tally_climbing = tally_gain >= delta_pp
        reasons.append(
            f"tally acc H={hs[-2]}→{hs[-1]} Δ={tally_gain:+.2f} pp "
            f"({'climbing' if tally_climbing else 'plateau'})"
        )

        gap_hs = [h for h in hs if h in float_by_h]
        gaps = [_pp(tally_by_hs[(h, primary_s)] - float_by_h[h]) for h in gap_hs]
        if len(gaps) >= 2:
            dgap = gaps[-1] - gaps[-2]
            gap_closed_last = dgap >= delta_pp
            reasons.append(
                f"gap H={gap_hs[-2]}→{gap_hs[-1]} Δ={dgap:+.2f} pp "
                f"({'closing' if gap_closed_last else 'not closing'})"
            )
        if len(gaps) >= 3:
            d1 = abs(gaps[-1] - gaps[-2])
            d2 = abs(gaps[-2] - gaps[-3])
            gap_plateau = d1 < delta_pp and d2 < delta_pp
            reasons.append(
                f"gap last two |Δ|={d2:.2f}, {d1:.2f} pp "
                f"({'plateau' if gap_plateau else 'still moving'})"
            )
        else:
            reasons.append("need three H with float+tally to call a gap plateau")

        h_done = (not tally_climbing) and gap_plateau
        cand = hs[-1] * 2
        time_blocked = (
            longest_cell_sec is not None and longest_cell_sec > cell_sec_cap
        )
        if time_blocked:
            reasons.append(
                f"cell wall {longest_cell_sec / 3600:.1f} h > "
                f"{cell_sec_cap / 3600:.0f} h cap"
            )
        if cand > h_cap:
            reasons.append(f"H cap {h_cap} (next would be {cand})")
            if not h_done:
                reasons.append("width still open but H cap hit → inconclusive/cap")
        elif time_blocked and not h_done:
            reasons.append("width still open but cell-time cap hit → inconclusive/cap")
        elif not h_done:
            expand_h = True
            next_h = cand
        else:
            reasons.append(f"width + gap plateau at H={hs[-1]}")

    # S search H: largest H if width still growing, else Tally-acc peak
    s_h: int | None = None
    if hs:
        if expand_h:
            s_h = hs[-1]
        else:
            s_h = max(hs, key=lambda h: tally_by_hs[(h, primary_s)])

    expand_s = False
    next_s_values: tuple[int, ...] = ()
    if s_h is None:
        reasons.append("no tally H to attach an S search")
    else:
        s_vals = sorted({s for (h, s) in tally_by_hs if h == s_h})
        missing = tuple(s for s in S_CROSS if s not in s_vals)
        if missing:
            expand_s = True
            next_s_values = missing
            reasons.append(
                f"S cross incomplete at H={s_h} (have {s_vals}, missing {list(missing)})"
            )
        elif len(s_vals) < 2:
            reasons.append(f"only S={s_vals} at H={s_h}; need a neighbor to judge S")
        else:
            s_max = s_vals[-1]
            prev = s_max // 2 if (s_max // 2) in s_vals else s_vals[-2]
            gain = _pp(tally_by_hs[(s_h, s_max)] - tally_by_hs[(s_h, prev)])
            s_climbing = gain >= delta_pp
            gap_closed_s = False
            if s_h in float_by_h:
                g_max = _pp(tally_by_hs[(s_h, s_max)] - float_by_h[s_h])
                g_prev = _pp(tally_by_hs[(s_h, prev)] - float_by_h[s_h])
                gap_closed_s = (g_max - g_prev) >= delta_pp
            reasons.append(
                f"tally acc S={prev}→{s_max} at H={s_h} Δ={gain:+.2f} pp "
                f"({'climbing' if s_climbing else 'plateau'})"
            )
            cand_s = s_max * 2
            if cand_s > s_cap:
                reasons.append(f"S cap {s_cap} (next would be {cand_s})")
                if s_climbing or gap_closed_s:
                    reasons.append("S still open but S cap hit → inconclusive/cap")
            elif s_climbing or gap_closed_s:
                expand_s = True
                next_s_values = (cand_s,)
            else:
                reasons.append(f"S plateau at S={s_max} (H={s_h})")

    stop = (not expand_h) and (not expand_s)
    return ExpandDecision(
        expand_h=expand_h,
        next_h=next_h,
        expand_s=expand_s,
        next_s_values=next_s_values,
        s_h=s_h,
        stop=stop,
        reasons=tuple(reasons),
    )
