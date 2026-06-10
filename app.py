"""
ML Exclusion FVA Analyzer
=========================
Upload a forecast extract at ANY level (SKU, SPU, CVC, country, ...) and judge
whether each ML exclusion is justified, using Forecast Value Added (FVA) logic:

    An exclusion is justified only if the user/consensus forecast beats the
    ML engine's own (shadow) forecast against shipped actuals.

Expected data shape (long format): one row per item per month, with
  - one or more dimension columns (any level you like)
  - an ML exclusion flag (Y/N)
  - a month/period column
  - ML forecast, user/consensus forecast, shipped units (actuals)

Column names are auto-detected but fully re-mappable in the sidebar.
"""

import io

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ----------------------------------------------------------------------------- 
# Page config & styling
# -----------------------------------------------------------------------------
st.set_page_config(page_title="ML Exclusion FVA Analyzer", page_icon="🎯",
                   layout="wide")

st.markdown("""
<style>
    .block-container {padding-top: 2rem;}
    div[data-testid="stMetricValue"] {font-size: 1.6rem;}
    .verdict-green  {color:#16a34a; font-weight:600;}
    .verdict-red    {color:#dc2626; font-weight:600;}
    .verdict-amber  {color:#d97706; font-weight:600;}
</style>
""", unsafe_allow_html=True)

st.title("🎯 ML Exclusion FVA Analyzer")
st.caption(
    "Judge whether ML exclusions are justified: does the human forecast "
    "actually beat what the ML engine would have produced, measured against "
    "shipped units?"
)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def guess_column(columns, keywords):
    """Return first column whose lowercase name contains any keyword."""
    for col in columns:
        low = col.lower()
        for kw in keywords:
            if kw in low:
                return col
    return None


@st.cache_data(show_spinner=False)
def load_file(file_bytes, file_name):
    if file_name.lower().endswith((".xlsx", ".xlsm", ".xls")):
        df = pd.read_excel(io.BytesIO(file_bytes))
    else:
        df = pd.read_csv(io.BytesIO(file_bytes))
    df.columns = [str(c).strip() for c in df.columns]
    return df


def wmape(err_abs_sum, actual_sum):
    if actual_sum > 0:
        return err_abs_sum / actual_sum
    return np.nan


def fmt_pct(x):
    return "—" if pd.isna(x) else f"{x:0.0%}"


def make_label(frame, cols):
    """Concatenate dimension columns into one display label (version-proof)."""
    lbl = frame[cols[0]].astype(str)
    for c in cols[1:]:
        lbl = lbl + " | " + frame[c].astype(str)
    return lbl


# -----------------------------------------------------------------------------
# 1 · Upload
# -----------------------------------------------------------------------------
uploaded = st.file_uploader(
    "Upload your monthly extract (CSV or Excel, long format: one row per item per month)",
    type=["csv", "xlsx", "xlsm", "xls"],
)

if uploaded is None:
    st.info(
        "**Waiting for a file.** Required content: dimension column(s) at any "
        "level, an ML-exclusion flag (Y/N), a month column, ML forecast, "
        "user/consensus forecast, and shipped units."
    )
    st.stop()

raw = load_file(uploaded.getvalue(), uploaded.name)
all_cols = list(raw.columns)

# -----------------------------------------------------------------------------
# 2 · Column mapping (auto-detected, user can override)
# -----------------------------------------------------------------------------
st.sidebar.header("1 · Column mapping")

# Sequential auto-detection: once a column is claimed, it can't be claimed again,
# and more specific keyword sets run first so "ML Exclusion" can't steal the
# "ML forecast" slot (or vice versa).
def guess_sequential(columns):
    taken, out = set(), {}
    specs = [
        ("excl",  ["exclusion", "excluded", "excl flag"]),
        ("month", ["month", "period", "fiscper", "date"]),
        ("act",   ["shipped", "actual", "sales qty", "deliver", "sales"]),
        ("ml",    ["ml forecast", "ml fc", "stat fc", "statistical", "engine",
                   "shadow"]),
        ("user",  ["consensus", "user forecast", "user fc", "final fc",
                   "demand plan", "adopted", "pbu"]),
        # last-resort loose passes
        ("ml",    ["ml"]),
        ("user",  ["user", "final"]),
    ]
    for slot, kws in specs:
        if slot in out:
            continue
        for col in columns:
            if col in taken:
                continue
            low = col.lower()
            if any(kw in low for kw in kws):
                out[slot] = col
                taken.add(col)
                break
    return out


guesses = guess_sequential(all_cols)
g_excl = guesses.get("excl")
g_month = guesses.get("month")
g_ml = guesses.get("ml")
g_user = guesses.get("user")
g_act = guesses.get("act")


def sel(label, guess, key):
    options = all_cols
    idx = options.index(guess) if guess in options else 0
    return st.sidebar.selectbox(label, options, index=idx, key=key)


col_excl = sel("ML exclusion flag (Y/N)", g_excl, "c_excl")
col_month = sel("Month / period", g_month, "c_month")
col_ml = sel("ML forecast (shadow / engine output)", g_ml, "c_ml")
col_user = sel("User / consensus forecast", g_user, "c_user")
col_act = sel("Shipped units (actuals)", g_act, "c_act")

# ---- Mapping validation: never run silently on a broken mapping -------------
mapped = [col_excl, col_month, col_ml, col_user, col_act]
if len(set(mapped)) < len(mapped):
    st.error(
        "⛔ **Same column mapped to two roles.** Each of the five roles in the "
        "sidebar (exclusion flag, month, ML forecast, user forecast, shipped "
        "units) must point at a different column. Fix the mapping to continue."
    )
    st.stop()

_flag_vals = set(raw[col_excl].dropna().astype(str).str.strip().str.upper().unique())
_yes_vals = {"Y", "YES", "TRUE", "1", "X"}
if not (_flag_vals & _yes_vals):
    st.warning(
        f"⚠️ The exclusion-flag column **{col_excl}** contains no Y/Yes/True/1/X "
        f"values — every item will be treated as not excluded. Values found: "
        f"`{sorted(_flag_vals)[:10]}`. If this is the wrong column, fix the "
        "mapping in the sidebar."
    )

for _role, _c in [("ML forecast", col_ml), ("User forecast", col_user),
                  ("Shipped units", col_act)]:
    _num = pd.to_numeric(raw[_c], errors="coerce")
    if _num.notna().mean() < 0.5:
        st.warning(
            f"⚠️ **{_role}** is mapped to `{_c}`, but most of its values are "
            "not numeric — this looks like a wrong mapping."
        )

with st.expander("🧭 Column mapping in use", expanded=False):
    st.table(pd.DataFrame({
        "Role": ["Exclusion flag", "Month", "ML forecast", "User forecast",
                 "Shipped units"],
        "Column": mapped,
        "Sample": [str(raw[c].dropna().iloc[0]) if raw[c].notna().any() else "—"
                   for c in mapped],
    }))

measure_cols = {col_month, col_ml, col_user, col_act, col_excl}
dim_candidates = [c for c in all_cols if c not in measure_cols]

st.sidebar.header("2 · Analysis level")
level_cols = st.sidebar.multiselect(
    "Group results by (pick the level you want verdicts at)",
    dim_candidates,
    default=dim_candidates,
    help="E.g. SKU+SPU for CVC level, only SKU for SKU level, or a country "
         "column for country level. Measures are summed to this level first.",
)
if not level_cols:
    st.warning("Pick at least one dimension column to analyse on.")
    st.stop()

# -----------------------------------------------------------------------------
# 3 · Settings
# -----------------------------------------------------------------------------
st.sidebar.header("3 · Verdict settings")

# ---- Completed months only: drop the current month and anything beyond ------
def parse_periods(vals):
    """Try to interpret period values as calendar months. Returns parsed
    datetimes aligned to vals, or None if the values carry no calendar info
    (e.g. relative period numbers 1..12)."""
    s = pd.Series(list(vals))
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().all():
        if num.between(1, 600).all():
            return None  # relative period numbering, no calendar anchor
        if num.between(190001, 209912).all():  # YYYYMM
            return pd.to_datetime(num.astype(int).astype(str), format="%Y%m",
                                  errors="coerce")
    try:
        parsed = pd.to_datetime(s.astype(str), format="mixed",
                                dayfirst=True, errors="coerce")
    except (TypeError, ValueError):
        parsed = pd.to_datetime(s.astype(str), dayfirst=True, errors="coerce")
    return parsed if parsed.notna().mean() >= 0.9 else None


_period_vals = list(raw[col_month].dropna().unique())
_parsed = parse_periods(_period_vals)
if _parsed is not None:
    _cutoff = pd.Timestamp.today().normalize().replace(day=1)  # 1st of current month
    _keep = {v for v, p in zip(_period_vals, _parsed)
             if pd.notna(p) and p < _cutoff}
    _dropped = [v for v in _period_vals if v not in _keep]
    if _dropped:
        raw = raw[raw[col_month].isin(_keep)]
        st.info(
            f"🗓️ **{len(_dropped)} period(s) excluded from analysis** — only "
            f"completed months are scored (today is "
            f"{pd.Timestamp.today():%d %b %Y}, so data up to "
            f"{(_cutoff - pd.Timedelta(days=1)):%b %Y} is used). Dropped: "
            f"{', '.join(str(d) for d in sorted(_dropped)[:8])}"
            f"{'…' if len(_dropped) > 8 else ''}. Forecasts against months "
            "that haven't finished shipping would count as pure error and "
            "poison every accuracy figure."
        )
    if raw.empty:
        st.error("No completed months left after the cutoff — the file only "
                 "contains current/future periods.")
        st.stop()
else:
    _drop_n = st.sidebar.number_input(
        "Drop trailing period(s) as incomplete", min_value=0, max_value=6,
        value=0,
        help="Your period column has no calendar information (e.g. 1, 2, 3 …), "
             "so the app can't auto-detect the current month. If the last "
             "period(s) in the file are still shipping, drop them here.")
    if _drop_n:
        _keep_periods = sorted(raw[col_month].dropna().unique())[:-_drop_n]
        raw = raw[raw[col_month].isin(_keep_periods)]
        st.info(f"🗓️ Last {_drop_n} period(s) dropped as incomplete.")

months_all = sorted(raw[col_month].dropna().unique().tolist())
months_sorted = months_all[-12:]  # always cap analysis at the last 12 months
if len(months_all) > 12:
    st.sidebar.caption(
        f"ℹ️ File contains {len(months_all)} periods — analysis automatically "
        f"limited to the most recent 12 ({months_sorted[0]} → {months_sorted[-1]})."
    )
window = st.sidebar.slider(
    "Trailing window (most recent months used for the verdict)",
    min_value=1, max_value=len(months_sorted), value=len(months_sorted),
    help="Score only the last N months, so recently fixed exclusions aren't "
         "punished for old behaviour.",
)
fva_threshold = st.sidebar.slider(
    "FVA materiality threshold (pp of WMAPE)", 1, 25, 5,
    help="How much better one side must be before it counts as a clear win "
         "rather than a coin-flip.",
) / 100.0
min_volume = st.sidebar.number_input(
    "Ignore items with total shipped units below", min_value=0, value=0,
    help="Filters out near-zero-volume noise from the verdict table.",
)

# -----------------------------------------------------------------------------
# 4 · Prepare data
# -----------------------------------------------------------------------------
df = raw.copy()
for c in [col_ml, col_user, col_act]:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

df["_excl"] = (
    df[col_excl].astype(str).str.strip().str.upper()
    .map(lambda v: "Y" if v in ("Y", "YES", "TRUE", "1", "X") else "N")
)

window_months = months_sorted[-window:]
dfw = df[df[col_month].isin(window_months)].copy()

# Aggregate measures to chosen level + month
grp_month = (
    dfw.groupby(level_cols + [col_month], dropna=False)
       .agg(ml=(col_ml, "sum"), user=(col_user, "sum"), act=(col_act, "sum"),
            excl=("_excl", lambda s: "Y" if (s == "Y").all()
                  else ("N" if (s == "N").all() else "Mixed")))
       .reset_index()
)
grp_month["err_ml"] = (grp_month["ml"] - grp_month["act"]).abs()
grp_month["err_user"] = (grp_month["user"] - grp_month["act"]).abs()
grp_month["user_wins"] = grp_month["err_user"] < grp_month["err_ml"]
grp_month["ml_wins"] = grp_month["err_ml"] < grp_month["err_user"]

# Roll up to item level
item = (
    grp_month.groupby(level_cols, dropna=False)
    .agg(excl=("excl", lambda s: s.mode().iat[0] if not s.mode().empty else "Mixed"),
         months=("act", "size"),
         act_sum=("act", "sum"),
         ml_sum=("ml", "sum"),
         user_sum=("user", "sum"),
         err_ml_sum=("err_ml", "sum"),
         err_user_sum=("err_user", "sum"),
         user_win_n=("user_wins", "sum"),
         ml_win_n=("ml_wins", "sum"))
    .reset_index()
)

item["wmape_ml"] = item.apply(lambda r: wmape(r.err_ml_sum, r.act_sum), axis=1)
item["wmape_user"] = item.apply(lambda r: wmape(r.err_user_sum, r.act_sum), axis=1)
# FVA: positive = the human/user forecast adds value over ML
item["fva"] = item["wmape_ml"] - item["wmape_user"]
item["bias_ml"] = np.where(item["act_sum"] > 0,
                           (item["ml_sum"] - item["act_sum"]) / item["act_sum"],
                           np.nan)
item["bias_user"] = np.where(item["act_sum"] > 0,
                             (item["user_sum"] - item["act_sum"]) / item["act_sum"],
                             np.nan)
item["user_win_rate"] = item["user_win_n"] / item["months"]

# Shadow-ML detection per item: months with demand where ML == user exactly
shadow = (
    grp_month.assign(identical=lambda d: (d["ml"] == d["user"]) &
                     ((d["act"] > 0) | (d["ml"] > 0) | (d["user"] > 0)))
    .groupby(level_cols)["identical"].mean().rename("identical_share")
    .reset_index()
)
item = item.merge(shadow, on=level_cols, how="left")


def verdict(row):
    if row["act_sum"] < min_volume:
        return "⚪ Below volume floor"
    if pd.isna(row["fva"]):
        return "⚪ No demand in window"
    if row["excl"] == "Y" and row["identical_share"] >= 0.8:
        return "⚠️ Shadow-ML missing"
    clear_user = row["fva"] >= fva_threshold and row["user_win_rate"] >= 0.55
    clear_ml = row["fva"] <= -fva_threshold and row["user_win_rate"] <= 0.45
    if row["excl"] == "Y":
        if clear_user:
            return "🟢 Exclusion justified"
        if clear_ml:
            return "🔴 Exclusion NOT justified"
        return "🟡 Inconclusive"
    else:  # not excluded
        if clear_user:
            return "🔵 Consider excluding"
        return "🟢 ML doing fine"


item["verdict"] = item.apply(verdict, axis=1)

# -----------------------------------------------------------------------------
# 5 · Data-quality panel (the silent killers, surfaced)
# -----------------------------------------------------------------------------
excl_rows = dfw[dfw["_excl"] == "Y"]
nonzero_excl = excl_rows[(excl_rows[col_ml] != 0) | (excl_rows[col_user] != 0)]
ident = (nonzero_excl[col_ml] == nonzero_excl[col_user]).mean() if len(nonzero_excl) else np.nan

with st.expander("🔍 Data-quality checks (read this before trusting verdicts)",
                 expanded=bool(pd.notna(ident) and ident > 0.5)):
    c1, c2, c3 = st.columns(3)
    c1.metric("Excluded rows where ML = user forecast", fmt_pct(ident),
              help="On excluded items the adopted forecast IS the user "
                   "forecast. If the ML column merely echoes it, the engine's "
                   "own number was not captured and the comparison is blind "
                   "exactly where it matters (the shadow-ML problem).")
    zero_share = (dfw[col_act] == 0).mean()
    c2.metric("Months with zero shipments", fmt_pct(zero_share),
              help="High intermittency makes percentage errors unstable; "
                   "verdicts lean on volume-weighted WMAPE to compensate.")
    c3.metric("Months in window", f"{len(window_months)} of {len(months_sorted)}")
    if pd.notna(ident) and ident > 0.5:
        st.error(
            "More than half of the non-zero excluded rows have an identical ML "
            "and user forecast. The ML column likely contains the *adopted* "
            "forecast, not the engine's shadow forecast — verdicts on those "
            "items will read as coin-flips. Fix the extract (separate shadow-ML "
            "key figure in IBP) before acting on red lights."
        )

# -----------------------------------------------------------------------------
# 6 · Headline KPIs
# -----------------------------------------------------------------------------
excl_items = item[item["excl"] == "Y"]
n_excl = len(excl_items)
n_bad = (excl_items["verdict"] == "🔴 Exclusion NOT justified").sum()
n_good = (excl_items["verdict"] == "🟢 Exclusion justified").sum()
n_shadow = (excl_items["verdict"] == "⚠️ Shadow-ML missing").sum()
bad_volume = excl_items.loc[
    excl_items["verdict"] == "🔴 Exclusion NOT justified", "act_sum"].sum()

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Items analysed", f"{len(item):,}")
k2.metric("Excluded items", f"{n_excl:,}")
k3.metric("Justified ✅", f"{n_good:,}")
k4.metric("Not justified ❌", f"{n_bad:,}",
          help="ML would have beaten the human consistently — removal candidates.")
k5.metric("Volume behind ❌", f"{bad_volume:,.0f}",
          help="Shipped units sitting on exclusions that ML would have "
               "forecast better.")
if n_shadow:
    st.warning(f"⚠️ {n_shadow} excluded item(s) could not be judged because ML "
               f"and user forecasts are identical (shadow-ML missing).")

# -----------------------------------------------------------------------------
# 7 · Tabs: verdict table · scatter · drill-in
# -----------------------------------------------------------------------------
tab_summary, tab_drill = st.tabs(["📊 Summary", "🔬 Item drill-in"])


with tab_summary:
    # ---- Whole-dataset scorecard, split by exclusion flag --------------------
    st.subheader("Whole-dataset scorecard")
    st.caption("Errors computed at the chosen analysis level per month, then "
               "summed over the window — last 12 months max. Split into "
               "excluded (Y) and not-excluded (N) items.")

    def scorecard_row(label, frame):
        a = frame["act"].sum()
        m = frame["ml"].sum()
        u = frame["user"].sum()
        em = frame["err_ml"].sum()
        eu = frame["err_user"].sum()
        wm_ml = wmape(em, a)
        wm_user = wmape(eu, a)
        return {
            "Scope": label,
            "ML FC": m,
            "User FC": u,
            "Shipped units": a,
            "Σ |ML − Act|": em,
            "Σ |User − Act|": eu,
            "FC Acc — ML": np.nan if pd.isna(wm_ml) else max(0.0, 1 - wm_ml),
            "FC Acc — User": np.nan if pd.isna(wm_user) else max(0.0, 1 - wm_user),
            "FC Bias — ML": (m - a) / a if a else np.nan,
            "FC Bias — User": (u - a) / a if a else np.nan,
            "FVA": (wm_ml - wm_user
                    if pd.notna(wm_ml) and pd.notna(wm_user) else np.nan),
        }

    rows = [scorecard_row("All items", grp_month),
            scorecard_row("Excluded (Y)", grp_month[grp_month["excl"] == "Y"]),
            scorecard_row("Not excluded (N)", grp_month[grp_month["excl"] == "N"])]
    if (grp_month["excl"] == "Mixed").any():
        rows.append(scorecard_row("Mixed flag", grp_month[grp_month["excl"] == "Mixed"]))
    sc = pd.DataFrame(rows)
    sc["Verdict"] = np.select(
        [sc["FVA"] >= fva_threshold, sc["FVA"] <= -fva_threshold],
        ["👤 User adds value", "🤖 ML more accurate"], default="≈ Tie")
    sc.loc[sc["FVA"].isna(), "Verdict"] = "⚪ No demand"

    st.dataframe(
        sc, use_container_width=True, hide_index=True,
        column_config={
            "FC Acc — ML": st.column_config.NumberColumn(
                format="percent", help="FA = 1 − WMAPE, floored at 0."),
            "FC Acc — User": st.column_config.NumberColumn(format="percent"),
            "FC Bias — ML": st.column_config.NumberColumn(
                format="percent",
                help="(Σ Forecast − Σ Actual) / Σ Actual. Positive = "
                     "over-forecast, negative = under-forecast."),
            "FC Bias — User": st.column_config.NumberColumn(format="percent"),
            "FVA": st.column_config.NumberColumn(
                format="percent",
                help="WMAPE(ML) − WMAPE(User); positive = user adds value."),
            "Shipped units": st.column_config.NumberColumn(format="%,.0f"),
            "ML FC": st.column_config.NumberColumn(
                format="%,.0f", help="Total ML forecast over the window."),
            "User FC": st.column_config.NumberColumn(
                format="%,.0f", help="Total user/consensus forecast over the window."),
            "Σ |ML − Act|": st.column_config.NumberColumn(format="%,.0f"),
            "Σ |User − Act|": st.column_config.NumberColumn(format="%,.0f"),
        },
    )

    # One headline verdict, focused on the side the tool exists to judge
    y_row = sc[sc["Scope"] == "Excluded (Y)"].iloc[0]
    if pd.isna(y_row["FVA"]):
        st.info("No shipped volume on excluded items in the window — no "
                "verdict on the exclusion list as a whole.")
    elif y_row["FVA"] >= fva_threshold:
        st.success(
            f"**Excluded items overall: 👤 the user forecast adds value** — "
            f"FC Acc {y_row['FC Acc — User']:.0%} vs {y_row['FC Acc — ML']:.0%} "
            f"for ML (FVA {y_row['FVA']:+.0%}). The exclusion list is earning "
            "its keep in aggregate; use the drill-in for item exceptions."
        )
    elif y_row["FVA"] <= -fva_threshold:
        st.error(
            f"**Excluded items overall: 🤖 ML would be more accurate** — "
            f"FC Acc {y_row['FC Acc — ML']:.0%} vs {y_row['FC Acc — User']:.0%} "
            f"for the user forecast (FVA {y_row['FVA']:+.0%}). The exclusion "
            "list as a whole is destroying forecast value — review it."
        )
    else:
        st.warning(
            f"**Excluded items overall: ≈ coin-flip** — FA difference "
            f"({y_row['FVA']:+.1%}) is below your {fva_threshold:.0%} "
            "materiality threshold."
        )

    # ---- Breakdown by any column, split by exclusion flag --------------------
    st.subheader("Summed-up analysis by…")
    bd_options = [c for c in all_cols
                  if c not in {col_ml, col_user, col_act}]
    default_bd = col_excl
    bcol = st.selectbox("Break the scorecard down by", bd_options,
                        index=bd_options.index(default_bd))

    # Group by the chosen column AND the exclusion flag, so both sides
    # (excluded Y vs not-excluded N) are visible within every group.
    split_by_excl = bcol != col_excl
    keys = [bcol, "_excl"] if split_by_excl else [bcol]

    bg = (
        dfw.groupby(keys + [col_month], dropna=False)
           .agg(ml=(col_ml, "sum"), user=(col_user, "sum"),
                act=(col_act, "sum"))
           .reset_index()
    )
    bg["err_ml"] = (bg["ml"] - bg["act"]).abs()
    bg["err_user"] = (bg["user"] - bg["act"]).abs()
    bsum = (
        bg.groupby(keys, dropna=False)
          .agg(act=("act", "sum"), ml_sum=("ml", "sum"),
               user_sum=("user", "sum"), err_ml=("err_ml", "sum"),
               err_user=("err_user", "sum"))
          .reset_index()
    )
    bsum["bias_ml"] = np.where(bsum["act"] > 0,
                               (bsum["ml_sum"] - bsum["act"]) / bsum["act"],
                               np.nan)
    bsum["bias_user"] = np.where(bsum["act"] > 0,
                                 (bsum["user_sum"] - bsum["act"]) / bsum["act"],
                                 np.nan)
    bsum["wmape_ml"] = np.where(bsum["act"] > 0,
                                bsum["err_ml"] / bsum["act"], np.nan)
    bsum["wmape_user"] = np.where(bsum["act"] > 0,
                                  bsum["err_user"] / bsum["act"], np.nan)
    bsum["fa_ml"] = (1 - bsum["wmape_ml"]).clip(lower=0)
    bsum["fa_user"] = (1 - bsum["wmape_user"]).clip(lower=0)
    bsum["fva"] = bsum["wmape_ml"] - bsum["wmape_user"]
    bsum["Verdict"] = np.select(
        [bsum["fva"] >= fva_threshold, bsum["fva"] <= -fva_threshold],
        ["👤 User adds value", "🤖 ML more accurate"], default="≈ Tie")
    bsum.loc[bsum["fva"].isna(), "Verdict"] = "⚪ No demand"
    # Sort: biggest groups first, Y before N within each group
    if split_by_excl:
        vol_order = (bsum.groupby(bcol)["act"].sum()
                         .sort_values(ascending=False).index.tolist())
        bsum["_vol_rank"] = bsum[bcol].map({v: i for i, v in enumerate(vol_order)})
        bsum = bsum.sort_values(["_vol_rank", "_excl"],
                                ascending=[True, False]).drop(columns="_vol_rank")
    else:
        bsum = bsum.sort_values("act", ascending=False)

    show_cols = [str(bcol)] + (["Excluded"] if split_by_excl else []) + [
        "Shipped units", "Σ |ML − Act|", "Σ |User − Act|",
        "FC Acc — ML", "FC Acc — User", "FC Bias — ML", "FC Bias — User",
        "FVA", "Verdict"]
    st.dataframe(
        bsum.rename(columns={
            bcol: str(bcol), "_excl": "Excluded", "act": "Shipped units",
            "err_ml": "Σ |ML − Act|", "err_user": "Σ |User − Act|",
            "fa_ml": "FC Acc — ML", "fa_user": "FC Acc — User",
            "bias_ml": "FC Bias — ML", "bias_user": "FC Bias — User",
            "fva": "FVA"})[show_cols],
        use_container_width=True, hide_index=True,
        column_config={
            "FC Acc — ML": st.column_config.NumberColumn(format="percent"),
            "FC Acc — User": st.column_config.NumberColumn(format="percent"),
            "FC Bias — ML": st.column_config.NumberColumn(
                format="percent",
                help="Positive = over-forecast, negative = under-forecast."),
            "FC Bias — User": st.column_config.NumberColumn(format="percent"),
            "FVA": st.column_config.NumberColumn(
                format="percent",
                help="WMAPE(ML) − WMAPE(User); positive = user adds value."),
            "Shipped units": st.column_config.NumberColumn(format="%,.0f"),
            "Σ |ML − Act|": st.column_config.NumberColumn(format="%,.0f"),
            "Σ |User − Act|": st.column_config.NumberColumn(format="%,.0f"),
        },
    )

    if split_by_excl:
        top_groups = (bsum.groupby(bcol)["act"].sum()
                          .sort_values(ascending=False).head(12).index.tolist())
        plot_bd = bsum[bsum[bcol].isin(top_groups) & bsum["fa_ml"].notna()]
        if not plot_bd.empty:
            long = plot_bd.melt(
                id_vars=[bcol, "_excl"], value_vars=["fa_ml", "fa_user"],
                var_name="side", value_name="fa")
            long["side"] = long["side"].map({"fa_ml": "FC Acc — ML",
                                             "fa_user": "FC Acc — User"})
            figb = px.bar(
                long, x=bcol, y="fa", color="side", barmode="group",
                facet_col="_excl",
                color_discrete_map={"FC Acc — ML": "#ef4444",
                                    "FC Acc — User": "#3b82f6"},
                category_orders={bcol: [str(g) for g in top_groups],
                                 "_excl": ["Y", "N", "Mixed"]},
                labels={"fa": "Forecast Accuracy", "side": ""},
            )
            figb.for_each_annotation(lambda a: a.update(
                text="Excluded (Y)" if a.text.endswith("Y")
                else ("Not excluded (N)" if a.text.endswith("N") else a.text)))
            figb.update_layout(height=440, yaxis_tickformat=".0%",
                               legend=dict(orientation="h", y=1.12))
            figb.update_yaxes(tickformat=".0%")
            st.plotly_chart(figb, use_container_width=True)
            st.caption("Top 12 groups by shipped volume, split into excluded "
                       "(Y) and not-excluded (N) items.")
    else:
        plot_bd = bsum[bsum["fa_ml"].notna()].head(20)
        if len(plot_bd) > 1:
            figb = go.Figure()
            figb.add_bar(x=plot_bd[bcol].astype(str), y=plot_bd["fa_ml"],
                         name="FC Acc — ML", marker_color="#ef4444")
            figb.add_bar(x=plot_bd[bcol].astype(str), y=plot_bd["fa_user"],
                         name="FC Acc — User", marker_color="#3b82f6")
            figb.update_layout(barmode="group", height=420,
                               yaxis_tickformat=".0%",
                               yaxis_title="Forecast Accuracy",
                               xaxis_title=str(bcol),
                               legend=dict(orientation="h", y=1.1))
            figb.update_xaxes(categoryorder="array",
                              categoryarray=plot_bd[bcol].astype(str).tolist())
            st.plotly_chart(figb, use_container_width=True)
            st.caption("Top 20 groups by shipped volume.")

    export_cols = level_cols + ["excl", "verdict", "fva", "wmape_ml",
                                "wmape_user", "user_win_rate", "bias_user",
                                "bias_ml", "act_sum", "months"]
    st.download_button(
        "⬇️ Download item-level verdicts (CSV)",
        item.sort_values("fva")[export_cols].to_csv(index=False).encode(),
        "exclusion_verdicts.csv", "text/csv",
        help="Full per-item verdict table at the chosen analysis level.")

with tab_drill:
    item["_label"] = make_label(item, level_cols)
    order = item.sort_values("fva")["_label"].tolist()
    pick = st.selectbox("Pick an item (sorted worst FVA first)", order)
    sel_row = item[item["_label"] == pick].iloc[0]
    mask = pd.Series(True, index=grp_month.index)
    for c in level_cols:
        mask &= grp_month[c].astype(str) == str(sel_row[c])
    ts = grp_month[mask].sort_values(col_month)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Verdict", sel_row["verdict"])
    m2.metric("FVA", fmt_pct(sel_row["fva"]))
    m3.metric("User win rate",
              f"{sel_row['user_win_n']:.0f}/{sel_row['months']:.0f} months")
    m4.metric("Shipped units", f"{sel_row['act_sum']:,.0f}")

    fig2 = go.Figure()
    fig2.add_bar(x=ts[col_month], y=ts["act"], name="Shipped units",
                 marker_color="#94a3b8", opacity=0.55)
    fig2.add_scatter(x=ts[col_month], y=ts["ml"], name="ML forecast",
                     mode="lines+markers", line=dict(color="#ef4444", width=2))
    fig2.add_scatter(x=ts[col_month], y=ts["user"], name="User forecast",
                     mode="lines+markers", line=dict(color="#3b82f6", width=2))
    fig2.update_layout(height=440, xaxis_title="Month", yaxis_title="Units",
                       legend=dict(orientation="h", y=1.08))
    st.plotly_chart(fig2, use_container_width=True)

    month_tbl = ts[[col_month, "ml", "user", "act", "err_ml", "err_user"]].copy()
    month_tbl["Closer"] = np.select(
        [month_tbl["err_user"] < month_tbl["err_ml"],
         month_tbl["err_ml"] < month_tbl["err_user"]],
        ["👤 User", "🤖 ML"], default="— Tie")
    st.dataframe(month_tbl.rename(columns={
        "ml": "ML FC", "user": "User FC", "act": "Shipped",
        "err_ml": "|ML err|", "err_user": "|User err|"}),
        use_container_width=True, hide_index=True)

st.caption(
    "Methodology: volume-weighted WMAPE per item over the trailing window; "
    "FVA = WMAPE(ML) − WMAPE(User), positive when the human adds value. "
    "Verdicts require both a material FVA gap and a consistent monthly win "
    "rate, so single-month flukes don't trigger lights. Verdicts are review "
    "candidates, not auto-decisions — the planner may know things the data "
    "doesn't."
)
