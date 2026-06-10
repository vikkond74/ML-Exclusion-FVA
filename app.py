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
tab_summary, tab_table, tab_scatter, tab_drill = st.tabs(
    ["📊 Summary", "📋 Verdict table", "📈 ML vs User error map",
     "🔬 Item drill-in"])

display_cols = level_cols + ["excl", "verdict", "fva", "wmape_ml",
                             "wmape_user", "user_win_rate", "bias_user",
                             "bias_ml", "act_sum", "months"]

with tab_summary:
    # ---- Whole-dataset scorecard --------------------------------------------
    st.subheader("Whole-dataset scorecard")
    st.caption("Errors computed at the chosen analysis level per month, then "
               "summed over the window — last 12 months max.")

    tot_act = grp_month["act"].sum()
    tot_err_ml = grp_month["err_ml"].sum()
    tot_err_user = grp_month["err_user"].sum()
    wm_ml_tot = wmape(tot_err_ml, tot_act)
    wm_user_tot = wmape(tot_err_user, tot_act)
    fa_ml_tot = np.nan if pd.isna(wm_ml_tot) else max(0.0, 1 - wm_ml_tot)
    fa_user_tot = np.nan if pd.isna(wm_user_tot) else max(0.0, 1 - wm_user_tot)
    fva_tot = (wm_ml_tot - wm_user_tot
               if pd.notna(wm_ml_tot) and pd.notna(wm_user_tot) else np.nan)

    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("Shipped units", f"{tot_act:,.0f}")
    s2.metric("Σ |ML − Actual|", f"{tot_err_ml:,.0f}")
    s3.metric("Σ |User − Actual|", f"{tot_err_user:,.0f}",
              delta=f"{tot_err_ml - tot_err_user:,.0f} vs ML",
              delta_color="normal")
    s4.metric("Forecast Accuracy — ML", fmt_pct(fa_ml_tot),
              help="FA = 1 − WMAPE, floored at 0.")
    s5.metric("Forecast Accuracy — User", fmt_pct(fa_user_tot))

    if pd.isna(fva_tot):
        st.info("Not enough shipped volume in the window for an overall verdict.")
    elif fva_tot >= fva_threshold:
        st.success(
            f"**Overall verdict: 👤 the user forecast adds value** — "
            f"FA {fmt_pct(fa_user_tot)} vs {fmt_pct(fa_ml_tot)} for ML "
            f"(FVA {fva_tot:+.0%})."
        )
    elif fva_tot <= -fva_threshold:
        st.error(
            f"**Overall verdict: 🤖 ML would be more accurate** — "
            f"FA {fmt_pct(fa_ml_tot)} vs {fmt_pct(fa_user_tot)} for the user "
            f"forecast (FVA {fva_tot:+.0%})."
        )
    else:
        st.warning(
            f"**Overall verdict: ≈ coin-flip** — FA difference "
            f"({fva_tot:+.1%}) is below your {fva_threshold:.0%} materiality "
            "threshold."
        )

    # ---- Breakdown by any column -------------------------------------------
    st.subheader("Summed-up analysis by…")
    bd_options = [c for c in all_cols
                  if c not in {col_ml, col_user, col_act}] 
    default_bd = col_excl
    bcol = st.selectbox("Break the scorecard down by", bd_options,
                        index=bd_options.index(default_bd))

    bg = (
        dfw.groupby([bcol, col_month], dropna=False)
           .agg(ml=(col_ml, "sum"), user=(col_user, "sum"),
                act=(col_act, "sum"))
           .reset_index()
    )
    bg["err_ml"] = (bg["ml"] - bg["act"]).abs()
    bg["err_user"] = (bg["user"] - bg["act"]).abs()
    bsum = (
        bg.groupby(bcol, dropna=False)
          .agg(act=("act", "sum"), err_ml=("err_ml", "sum"),
               err_user=("err_user", "sum"))
          .reset_index()
    )
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
    bsum = bsum.sort_values("act", ascending=False)

    st.dataframe(
        bsum.rename(columns={
            bcol: str(bcol), "act": "Shipped units",
            "err_ml": "Σ |ML − Act|", "err_user": "Σ |User − Act|",
            "fa_ml": "FA — ML", "fa_user": "FA — User", "fva": "FVA"})
        [[str(bcol), "Shipped units", "Σ |ML − Act|", "Σ |User − Act|",
          "FA — ML", "FA — User", "FVA", "Verdict"]],
        use_container_width=True, hide_index=True,
        column_config={
            "FA — ML": st.column_config.NumberColumn(format="percent"),
            "FA — User": st.column_config.NumberColumn(format="percent"),
            "FVA": st.column_config.NumberColumn(
                format="percent",
                help="WMAPE(ML) − WMAPE(User); positive = user adds value."),
            "Shipped units": st.column_config.NumberColumn(format="%,.0f"),
            "Σ |ML − Act|": st.column_config.NumberColumn(format="%,.0f"),
            "Σ |User − Act|": st.column_config.NumberColumn(format="%,.0f"),
        },
    )

    plot_bd = bsum[bsum["fa_ml"].notna()].head(20)
    if len(plot_bd) > 1:
        figb = go.Figure()
        figb.add_bar(x=plot_bd[bcol].astype(str), y=plot_bd["fa_ml"],
                     name="FA — ML", marker_color="#ef4444")
        figb.add_bar(x=plot_bd[bcol].astype(str), y=plot_bd["fa_user"],
                     name="FA — User", marker_color="#3b82f6")
        figb.update_layout(barmode="group", height=420,
                           yaxis_tickformat=".0%",
                           yaxis_title="Forecast Accuracy",
                           xaxis_title=str(bcol),
                           legend=dict(orientation="h", y=1.1))
        figb.update_xaxes(categoryorder="array",
                          categoryarray=plot_bd[bcol].astype(str).tolist())
        st.plotly_chart(figb, use_container_width=True)
        st.caption("Top 20 groups by shipped volume.")

with tab_table:
    f1, f2 = st.columns([1, 2])
    flag_filter = f1.multiselect("Exclusion flag", ["Y", "N", "Mixed"],
                                 default=["Y"])
    verdict_filter = f2.multiselect(
        "Verdict", sorted(item["verdict"].unique().tolist()), default=[])
    tbl = item[item["excl"].isin(flag_filter)] if flag_filter else item
    if verdict_filter:
        tbl = tbl[tbl["verdict"].isin(verdict_filter)]
    tbl = tbl.sort_values("fva")  # worst exclusions first
    if tbl.empty:
        st.info(
            "No rows match the current filters. If 'Excluded items' shows 0 "
            "above, either this extract truly has no exclusions or the "
            "exclusion-flag column is mis-mapped (check the sidebar). "
            "Otherwise, add 'N' to the Exclusion flag filter to see "
            "non-excluded items."
        )
    st.dataframe(
        tbl[display_cols].rename(columns={
            "excl": "Excluded", "verdict": "Verdict", "fva": "FVA",
            "wmape_ml": "WMAPE ML", "wmape_user": "WMAPE User",
            "user_win_rate": "User win rate", "bias_user": "Bias user",
            "bias_ml": "Bias ML", "act_sum": "Shipped units",
            "months": "Months"}),
        use_container_width=True, height=480,
        column_config={
            "FVA": st.column_config.NumberColumn(
                format="percent",
                help="WMAPE(ML) − WMAPE(User). Positive = human adds value."),
            "WMAPE ML": st.column_config.NumberColumn(format="percent"),
            "WMAPE User": st.column_config.NumberColumn(format="percent"),
            "User win rate": st.column_config.NumberColumn(format="percent"),
            "Bias user": st.column_config.NumberColumn(format="percent"),
            "Bias ML": st.column_config.NumberColumn(format="percent"),
        },
    )
    csv = tbl[display_cols].to_csv(index=False).encode()
    st.download_button("⬇️ Download verdict table (CSV)", csv,
                       "exclusion_verdicts.csv", "text/csv")

with tab_scatter:
    plot_df = item[(item["act_sum"] >= max(min_volume, 1)) &
                   item["wmape_ml"].notna() & item["wmape_user"].notna()].copy()
    if plot_df.empty:
        st.info("Nothing to plot at the current volume floor.")
    else:
        cap = st.slider("Cap WMAPE axis at", 0.5, 5.0, 2.0, 0.5)
        plot_df["wmape_ml_c"] = plot_df["wmape_ml"].clip(upper=cap)
        plot_df["wmape_user_c"] = plot_df["wmape_user"].clip(upper=cap)
        plot_df["label"] = make_label(plot_df, level_cols)
        fig = px.scatter(
            plot_df, x="wmape_ml_c", y="wmape_user_c", color="excl",
            size="act_sum", size_max=28, hover_name="label",
            hover_data={"fva": ":.0%", "user_win_rate": ":.0%",
                        "wmape_ml_c": False, "wmape_user_c": False},
            color_discrete_map={"Y": "#ef4444", "N": "#3b82f6",
                                "Mixed": "#a855f7"},
            labels={"wmape_ml_c": "WMAPE — ML forecast",
                    "wmape_user_c": "WMAPE — user forecast",
                    "excl": "Excluded"},
        )
        fig.add_shape(type="line", x0=0, y0=0, x1=cap, y1=cap,
                      line=dict(dash="dash", color="grey"))
        fig.add_annotation(x=cap * 0.78, y=cap * 0.94, showarrow=False,
                           text="Above line: ML wins → exclusion suspect",
                           font=dict(size=11, color="grey"))
        fig.add_annotation(x=cap * 0.22, y=cap * 0.04, showarrow=False,
                           text="Below line: human wins → exclusion justified",
                           font=dict(size=11, color="grey"))
        fig.update_layout(height=620)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Bubble size = shipped volume. Red bubbles **above** the "
                   "diagonal are excluded items where ML would have been more "
                   "accurate — the prime removal candidates.")

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
