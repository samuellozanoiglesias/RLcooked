#!/usr/bin/env python3
"""
paper_stats.py -- statistics and reporting conventions shared by every figure
script of the paper (main text and SI), so that all numbers are computed and
written in exactly the same way.

Conventions (those of the main-text human results):

  * Means are reported as mean +- SEM: game score with 2 decimals
    ($8.63 \\pm 0.38$), specialization index with 1 decimal and a percent sign
    ($59.7 \\pm 3.9\\%$).
  * Two groups are compared with Welch's t-test. The group with the HIGHER mean
    is written first and t is reported as a positive number; the direction is
    carried by the wording ("HA teams score significantly higher ...",
    "MA teams show significantly higher task partitioning ..."):
        ($8.63 \\pm 0.38$ vs.\\ $5.31 \\pm 0.25$; Welch's $t(120.1) = 7.31$, $p < .001$)
    The degrees of freedom are the Welch-Satterthwaite df with 1 decimal.
  * Two-way between-subjects ANOVAs use effect (sum-to-zero) coding and Type III
    sums of squares (identical to statsmodels anova_lm(..., typ=3) with C(x, Sum)),
    with partial eta squared:
        ($F(1,276) = 39.34$, $p < .001$, $\\eta_p^2 = .12$)
  * p-values without the leading zero, 3 decimals, and "p < .001" below .001.
  * One-sample t-tests (only used where a reference has no dispersion) follow the
    same wording: $t(df) = |t|$.

The signed statistic (first group minus second group) is kept in every returned
dict and in the CSV outputs, for anyone who needs it.
"""

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# descriptive
# ---------------------------------------------------------------------------
def clean(v):
    v = np.asarray(v, float)
    return v[np.isfinite(v)]


def describe(v) -> dict:
    v = clean(v)
    n = v.size
    if n == 0:
        return dict(n=0, mean=np.nan, sd=np.nan, sem=np.nan, median=np.nan)
    sd = v.std(ddof=1) if n > 1 else np.nan
    return dict(n=n, mean=float(v.mean()), sd=sd, sem=sd / np.sqrt(n) if n > 1 else np.nan,
                median=float(np.median(v)))


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------
def welch(a, b) -> dict:
    """Welch's t-test of a vs b. Returns t_signed (a - b), t (= |t|), df, p,
    the label of the higher group ('a' | 'b') and Hedges' g (a - b)."""
    a, b = clean(a), clean(b)
    out = dict(t_signed=np.nan, t=np.nan, df=np.nan, p=np.nan, higher=None, g=np.nan,
               n_a=a.size, n_b=b.size)
    if a.size < 2 or b.size < 2:
        return out
    v1, v2 = a.var(ddof=1) / a.size, b.var(ddof=1) / b.size
    if v1 + v2 == 0:
        return out
    t, p = stats.ttest_ind(a, b, equal_var=False)
    dof = (v1 + v2) ** 2 / (v1 ** 2 / (a.size - 1) + v2 ** 2 / (b.size - 1))
    sp = np.sqrt(((a.size - 1) * a.var(ddof=1) + (b.size - 1) * b.var(ddof=1)) / (a.size + b.size - 2))
    g = (a.mean() - b.mean()) / sp * (1 - 3 / (4 * (a.size + b.size) - 9)) if sp > 0 else np.nan
    out.update(t_signed=float(t), t=abs(float(t)), df=float(dof), p=float(p),
               higher="a" if a.mean() >= b.mean() else "b", g=float(g))
    return out


def one_sample(a, mu) -> dict:
    """One-sample t-test of a against a fixed value mu (a reference without dispersion)."""
    a = clean(a)
    out = dict(t_signed=np.nan, t=np.nan, df=np.nan, p=np.nan, higher=None)
    if a.size < 2 or not np.isfinite(mu):
        return out
    t, p = stats.ttest_1samp(a, mu)
    out.update(t_signed=float(t), t=abs(float(t)), df=float(a.size - 1), p=float(p),
               higher="a" if a.mean() >= mu else "ref")
    return out


def mann_whitney(a, b) -> float:
    a, b = clean(a), clean(b)
    if a.size < 2 or b.size < 2:
        return np.nan
    return float(stats.mannwhitneyu(a, b, alternative="two-sided")[1])


def holm(pvals):
    p = np.asarray(pvals, float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    m = int(ok.sum())
    if m == 0:
        return out
    idx = np.argsort(p[ok])
    adj = np.minimum(1, np.maximum.accumulate((m - np.arange(m)) * p[ok][idx]))
    tmp = np.empty(m)
    tmp[idx] = adj
    out[ok] = tmp
    return out


def _effect_code(col, levels):
    X = np.zeros((len(col), len(levels) - 1))
    for j, lev in enumerate(levels[:-1]):
        X[:, j] = (col == lev).astype(float)
    X[col == levels[-1], :] = -1.0
    return X


def anova_type3(df: pd.DataFrame, y: str, a: str, b: str) -> pd.DataFrame:
    """Two-way between-subjects ANOVA, effect coding, Type III SS.
    Rows: a, b, 'a x b' with SS, df, df_error, F, p, eta_p2."""
    d = df[[y, a, b]].dropna()
    yv = d[y].to_numpy(float)
    la, lb = sorted(d[a].unique()), sorted(d[b].unique())
    names = [a, b, f"{a} x {b}"]
    if len(la) < 2 or len(lb) < 2:
        return pd.DataFrame([dict(term=t, SS=np.nan, df=np.nan, df_error=np.nan, F=np.nan, p=np.nan,
                                  eta_p2=np.nan) for t in names])
    Xa, Xb = _effect_code(d[a].to_numpy(), la), _effect_code(d[b].to_numpy(), lb)
    Xab = np.column_stack([Xa[:, i] * Xb[:, j] for i in range(Xa.shape[1]) for j in range(Xb.shape[1])])
    blocks = dict(zip(names, (Xa, Xb, Xab)))
    one = np.ones(len(yv))
    X_full = np.column_stack([one, Xa, Xb, Xab])

    def rss(X):
        beta = np.linalg.lstsq(X, yv, rcond=None)[0]
        return float(np.sum((yv - X @ beta) ** 2))

    rss_full = rss(X_full)
    df_err = len(yv) - np.linalg.matrix_rank(X_full)
    rows = []
    for name, blk in blocks.items():
        dfn = blk.shape[1]
        if df_err <= 0 or rss_full <= 1e-12:
            rows.append(dict(term=name, SS=np.nan, df=dfn, df_error=df_err, F=np.nan, p=np.nan, eta_p2=np.nan))
            continue
        others = [one] + [v for k, v in blocks.items() if k != name]
        ss = rss(np.column_stack(others)) - rss_full
        F = (ss / dfn) / (rss_full / df_err)
        rows.append(dict(term=name, SS=ss, df=dfn, df_error=df_err, F=F,
                         p=float(stats.f.sf(F, dfn, df_err)), eta_p2=ss / (ss + rss_full)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# formatting (LaTeX fragments in the style of the main text)
# ---------------------------------------------------------------------------
def _nolead(x: float, d: int) -> str:
    s = f"{x:.{d}f}"
    return s.replace("0.", ".", 1) if s.startswith(("0.", "-0.")) else s


def fmt_p(p, sub: str = "") -> str:
    sym = f"p_{{\\text{{{sub}}}}}" if sub else "p"
    if p is None or not np.isfinite(p):
        return f"${sym} = $ n/a"
    return f"${sym} < .001$" if p < 0.001 else f"${sym} = {_nolead(p, 3)}$"


def fmt_ms(desc: dict, pct: bool) -> str:
    """$8.63 \\pm 0.38$  |  $59.7 \\pm 3.9\\%$"""
    if not desc or not np.isfinite(desc.get("mean", np.nan)):
        return "n/a"
    sem = desc["sem"] if np.isfinite(desc["sem"]) else 0.0
    return f"${desc['mean']:.1f} \\pm {sem:.1f}\\%$" if pct else f"${desc['mean']:.2f} \\pm {sem:.2f}$"


def fmt_t(res: dict, welch_label: bool = True) -> str:
    if not np.isfinite(res.get("t", np.nan)):
        return "t n/a"
    name = "Welch's " if welch_label else ""
    return f"{name}$t({res['df']:.1f}) = {res['t']:.2f}$, {fmt_p(res['p'])}"


def fmt_F(row) -> str:
    if not np.isfinite(row["F"]):
        return "F n/a"
    eta = f", $\\eta_p^2 = {_nolead(row['eta_p2'], 2)}$" if np.isfinite(row["eta_p2"]) else ""
    return f"$F({int(row['df'])},{int(row['df_error'])}) = {row['F']:.2f}$, {fmt_p(row['p'])}{eta}"


def fmt_compare(name_a: str, a, name_b: str, b, pct: bool, welch_label: bool = True) -> tuple[str, dict]:
    """Paper-style comparison: higher group first, |t|.
    -> ('MA > HA: ($6.96 \\pm 0.37$ vs.\\ $5.99 \\pm 0.36$; Welch's $t(137.8) = 1.90$, $p = .059$)', res)"""
    res = welch(a, b)
    da, db = describe(a), describe(b)
    if res["higher"] == "b":
        (name_a, da), (name_b, db) = (name_b, db), (name_a, da)
    txt = (f"{name_a} > {name_b}: ({fmt_ms(da, pct)} vs.\\ {fmt_ms(db, pct)}; "
           f"{fmt_t(res, welch_label)})")
    return txt, res


def fmt_compare_ref(name_a: str, a, name_ref: str, mu: float, pct: bool) -> tuple[str, dict]:
    """One-sample comparison with a dispersion-free reference value, same wording:
    'humans > pretrained: ($2.31$ vs.\\ $1.92 \\pm 0.02$; one-sample $t(2) = 24.53$, $p = .002$)'."""
    res = one_sample(a, mu)
    ms_a = fmt_ms(describe(a), pct)
    ref = (f"${mu:.1f}\\%$" if pct else f"${mu:.2f}$") if np.isfinite(mu) else "n/a"
    if res["higher"] == "ref":
        head, vals = f"{name_ref} > {name_a}", f"{ref} vs.\\ {ms_a}"
    else:
        head, vals = f"{name_a} > {name_ref}", f"{ms_a} vs.\\ {ref}"
    t = (f"one-sample $t({int(res['df'])}) = {res['t']:.2f}$, {fmt_p(res['p'])}"
         if np.isfinite(res["t"]) else "t n/a")
    return f"{head}: ({vals}; {t})", res
