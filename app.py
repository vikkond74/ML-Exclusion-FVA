"""
ML Exclusion FVA Analyzer — Simple edition
==========================================
Upload a forecast extract at any level and judge whether ML exclusions are
justified: does the user/consensus forecast beat the ML engine's own (shadow)
forecast against shipped actuals?

Simple methodology (deliberately):
    Variance = Forecast − Shipped          (signed, units)
    Bias %   = Variance / Shipped          (positive = over-forecast)
    FCA      = 1 − |Variance| / Shipped    (floored at 0)
    Verdict  = compare FCA of User vs ML
"""

import io

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="ML Exclusion FVA Analyzer", page_icon="🎯",
                   layout="wide")
st.title("🎯 ML Exclusion FVA Analyzer")
st.caption("Is each ML exclusion earning its keep? Variance = FC − Shipped · "
           "Bias % = Variance / Shipped · FCA = 1 − |Variance| / Shipped.")


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_file(file_bytes, file_name):
    if file_name.lower().endswith((".xlsx", ".xlsm", ".xls")):
        df = pd.read_excel(io.BytesIO(file_bytes))
    else:
        df = pd.read_csv(io.BytesIO(file_bytes))
    df.columns = [str(c).strip() for c in df.columns]
    return df


def as_str(series):
    """NA-safe string conversion (pandas 3.x astype(str) keeps NA)."""
    return series.astype(str).fillna("(blank)")


def parse_periods(vals):
    """Interpret period values as calendar months, tolerating MIXED types in
    one column. Returns parsed datetimes aligned to vals, or None if values
    carry no calendar info (e.g. relative period numbers 1..12)."""
    s = pd.Series(list(vals))
    num = pd.to_numeric(s, errors="coerce")
    if (num.notna().all() and num.between(1, 600).all()
            and (num % 1 == 0).all()):
        return None
    parsed = pd.Series(pd.NaT, index=s.index)
    is_yyyymm = num.notna() & num.between(190001, 209912)
    if is_yyyymm.any():
        parsed[is_yyyymm] = pd.to_datetime(
            num[is_yyyymm].astype(int).astype(str), format="%Y%m",
            errors="coerce")
    rest = parsed.isna()
    if rest.any():
        ext = s[rest].astype(str).str.strip().str.extract(
            r"^(?P<m>\d{1,2})[./\-](?P<y>\d{4})$")
        ok = ext["m"].notna() & ext["y"].notna()
        if ok.any():
            mm = ext.loc[ok, "m"].astype(int)
            idx = ext.index[ok][mm.between(1, 12)]
            parsed.loc[idx] = pd.to_datetime(
                ext.loc[idx, "y"] + "-" + ext.loc[idx, "m"].str.zfill(2),
                format="%Y-%m", errors="coerce")
    rest = parsed.isna()
    if rest.any():
        try:
            parsed[rest] = pd.to_datetime(s[rest].astype(str), format="mixed",
                                          dayfirst=True, errors="coerce")
        except (TypeError, ValueError):
            parsed[rest] = pd.to_datetime(s[rest].astype(str), dayfirst=True,
                                          errors="coerce")
    return parsed if parsed.notna().mean() >= 0.9 else None


def simple_metrics(frame):
    """Variance / Bias % / FCA for a frame with ml, user, act columns,
    summed over the frame."""
    a = frame["act"].sum()
    m = frame["ml"].sum()
    u = frame["user"].sum()
    var_ml = m - a
    var_user = u - a
    return {
        "ML FC": m, "User FC": u, "Shipped": a,
        "Variance ML": var_ml, "Variance User": var_user,
        "Bias % ML": var_ml / a if a else np.nan,
        "Bias % User": var_user / a if a else np.nan,
        "FCA ML": max(0.0, 1 - abs(var_ml) / a) if a else np.nan,
        "FCA User": max(0.0, 1 - abs(var_user) / a) if a else np.nan,
    }


def add_verdict(df_, thr):
    diff = df_["FCA User"] - df_["FCA ML"]
    df_["FCA User − ML"] = diff
    df_["Verdict"] = np.select(
        [diff >= thr, diff <= -thr],
        ["👤 User adds value", "🤖 ML more accurate"], default="≈ Tie")
    df_.loc[diff.isna(), "Verdict"] = "⚪ No demand"
    return df_


NUM_COLS = ["ML FC", "User FC", "Shipped", "Variance ML", "Variance User"]
PCT_COLS = ["Bias % ML", "Bias % User", "FCA ML", "FCA User", "FCA User − ML"]
COLUMN_CONFIG = {
    **{c: st.column_config.NumberColumn(format="%,.0f") for c in NUM_COLS},
    **{c: st.column_config.NumberColumn(format="percent") for c in PCT_COLS},
    "Variance ML": st.column_config.NumberColumn(
        format="%,.0f", help="ML FC − Shipped (positive = over-forecast)."),
    "Variance User": st.column_config.NumberColumn(
        format="%,.0f", help="User FC − Shipped (positive = over-forecast)."),
    "FCA User − ML": st.column_config.NumberColumn(
        format="percent", help="Positive = the user forecast adds value."),
}


# -----------------------------------------------------------------------------
# 1 · Upload
# -----------------------------------------------------------------------------
uploaded = st.file_uploader(
    "Upload your extract (CSV or Excel, long format: one row per item per month)",
    type=["csv", "xlsx", "xlsm", "xls"],
)
if uploaded is None:
    st.info("**Waiting for a file.** Required content: dimension column(s) at "
            "any level, an ML-exclusion flag (Y/N), a month column, ML "
            "forecast, user/consensus forecast, and shipped units.")
    st.stop()

raw = load_file(uploaded.getvalue(), uploaded.name)
all_cols = list(raw.columns)

# -----------------------------------------------------------------------------
# 2 · Column mapping
# -----------------------------------------------------------------------------
st.sidebar.header("1 · Column mapping")


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
        ("ml",    ["ml"]),
        ("user",  ["user", "final"]),
    ]
    for slot, kws in specs:
        if slot in out:
            continue
        for col in columns:
            if col in taken:
                continue
            if any(kw in col.lower() for kw in kws):
                out[slot] = col
                taken.add(col)
                break
    return out


guesses = guess_sequential(all_cols)


def sel(label, slot, key):
    guess = guesses.get(slot)
    idx = all_cols.index(guess) if guess in all_cols else 0
    return st.sidebar.selectbox(label, all_cols, index=idx, key=key)


col_excl = sel("ML exclusion flag (Y/N)", "excl", "c_excl")
col_month = sel("Month / period", "month", "c_month")
col_ml = sel("ML forecast (shadow / engine output)", "ml", "c_ml")
col_user = sel("User / consensus forecast", "user", "c_user")
col_act = sel("Shipped units (actuals)", "act", "c_act")

mapped = [col_excl, col_month, col_ml, col_user, col_act]
if len(set(mapped)) < len(mapped):
    st.error("⛔ **Same column mapped to two roles.** Each of the five roles "
             "in the sidebar must point at a different column.")
    st.stop()

_flag_vals = set(raw[col_excl].dropna().astype(str).str.strip().str.upper()
                 .unique())
if not (_flag_vals & {"Y", "YES", "TRUE", "1", "X"}):
    st.warning(f"⚠️ The exclusion-flag column **{col_excl}** contains no "
               f"Y/Yes/True/1/X values — every item will be treated as not "
               f"excluded. Values found: `{sorted(_flag_vals)[:10]}`.")

for _role, _c in [("ML forecast", col_ml), ("User forecast", col_user),
                  ("Shipped units", col_act)]:
    if pd.to_numeric(raw[_c], errors="coerce").notna().mean() < 0.5:
        st.warning(f"⚠️ **{_role}** is mapped to `{_c}`, but most of its "
                   "values are not numeric — this looks like a wrong mapping.")

with st.expander("🧭 Column mapping in use", expanded=False):
    st.table(pd.DataFrame({
        "Role": ["Exclusion flag", "Month", "ML forecast", "User forecast",
                 "Shipped units"],
        "Column": mapped,
        "Sample": [str(raw[c].dropna().iloc[0]) if raw[c].notna().any()
                   else "—" for c in mapped],
    }))

# -----------------------------------------------------------------------------
# 3 · Period handling: normalize, completed months only, last-12 cap
# -----------------------------------------------------------------------------
_pvals = list(raw[col_month].dropna().unique())
_pnorm = parse_periods(_pvals)
if _pnorm is not None:
    _pmap = {v: (f"{d.year}-{d.month:02d}" if pd.notna(d) else None)
             for v, d in zip(_pvals, _pnorm)}
    _before = len(raw)
    raw = raw.copy()
    raw[col_month] = raw[col_month].map(_pmap)
    raw = raw[raw[col_month].notna()]
    if len(raw) < _before:
        st.warning(f"⚠️ {_before - len(raw)} row(s) dropped: period value "
                   "could not be interpreted as a month.")
else:
    _pnum_full = pd.to_numeric(raw[col_month], errors="coerce")
    if _pnum_full.notna().mean() >= 0.9:
        raw = raw[_pnum_full.notna()].copy()
        raw[col_month] = pd.to_numeric(raw[col_month])
    else:
        raw = raw.copy()
        raw[col_month] = raw[col_month].astype(str)

_period_vals = list(raw[col_month].dropna().unique())
_parsed = parse_periods(_period_vals)
if _parsed is not None:
    _cutoff = pd.Timestamp.today().normalize().replace(day=1)
    _keep = {v for v, p in zip(_period_vals, _parsed)
             if pd.notna(p) and p < _cutoff}
    _dropped = [v for v in _period_vals if v not in _keep]
    if _dropped:
        raw = raw[raw[col_month].isin(_keep)]
        st.info(f"🗓️ **{len(_dropped)} period(s) excluded** — only completed "
                f"months are scored (today is {pd.Timestamp.today():%d %b %Y})."
                f" Dropped: {', '.join(str(d) for d in sorted(_dropped)[:8])}"
                f"{'…' if len(_dropped) > 8 else ''}.")
    if raw.empty:
        st.error("No completed months left — the file only contains "
                 "current/future periods.")
        st.stop()
else:
    _nums = pd.to_numeric(pd.Series(_period_vals), errors="coerce")
    _cur_m = pd.Timestamp.today().month
    _cur_y = pd.Timestamp.today().year
    if (_nums.notna().all() and _nums.between(1, 12).all()
            and (_nums < _cur_m).all()):
        st.caption(f"🗓️ Period numbers {int(_nums.min())}–{int(_nums.max())} "
                   f"read as calendar months of {_cur_y}; all completed.")
    elif _nums.notna().all() and _nums.between(1, 12).all():
        interp = st.radio(
            f"⚠️ Month numbers reach {int(_nums.max())}, but only months "
            f"1–{_cur_m - 1} of {_cur_y} are completed. How should they be "
            "read?",
            [f"Calendar months of {_cur_y} → drop months ≥ {_cur_m} as "
             "incomplete",
             "Sequence numbers of past periods → keep all"], index=0)
        if interp.startswith("Calendar"):
            _keep_nums = {v for v, n in zip(_period_vals, _nums) if n < _cur_m}
            _dropped = [v for v in _period_vals if v not in _keep_nums]
            raw = raw[raw[col_month].isin(_keep_nums)]
            if _dropped:
                st.info(f"🗓️ Dropped month(s) "
                        f"{', '.join(str(d) for d in sorted(_dropped))} as "
                        "not yet completed.")
            if raw.empty:
                st.error("No completed months left after the cutoff.")
                st.stop()
    else:
        st.warning(f"⚠️ **Current/future months could NOT be auto-dropped** — "
                   f"the period column `{col_month}` carries no recognizable "
                   f"calendar information (values look like: "
                   f"`{', '.join(str(v) for v in _period_vals[:5])}`…). Use "
                   "the sidebar control to drop trailing incomplete periods.")
    _drop_n = st.sidebar.number_input(
        "Drop trailing period(s) as incomplete", min_value=0, max_value=6,
        value=0)
    if _drop_n:
        _keep_periods = sorted(raw[col_month].dropna().unique())[:-_drop_n]
        raw = raw[raw[col_month].isin(_keep_periods)]
        st.info(f"🗓️ Last {_drop_n} period(s) dropped as incomplete.")

st.sidebar.header("2 · Settings")
months_all = sorted(raw[col_month].dropna().unique().tolist())
months_sorted = months_all[-12:]
if len(months_all) > 12:
    st.sidebar.caption(f"ℹ️ {len(months_all)} periods in file — limited to "
                       f"the most recent 12.")
fca_threshold = st.sidebar.slider(
    "Verdict materiality threshold (pp of FCA)", 1, 25, 5,
    help="How much higher one side's FCA must be before it counts as a clear "
         "win rather than a tie.") / 100.0

# -----------------------------------------------------------------------------
# 4 · Prepare data
# -----------------------------------------------------------------------------
df = raw[raw[col_month].isin(months_sorted)].copy()
for c in [col_ml, col_user, col_act]:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
df["_excl"] = (df[col_excl].astype(str).str.strip().str.upper()
               .map(lambda v: "Y" if v in ("Y", "YES", "TRUE", "1", "X")
                    else "N"))
df = df.rename(columns={col_ml: "ml", col_user: "user", col_act: "act"})

# -----------------------------------------------------------------------------
# 5 · Data-quality checks
# -----------------------------------------------------------------------------
excl_rows = df[df["_excl"] == "Y"]
nonzero_excl = excl_rows[(excl_rows["ml"] != 0) | (excl_rows["user"] != 0)]
ident = ((nonzero_excl["ml"] == nonzero_excl["user"]).mean()
         if len(nonzero_excl) else np.nan)
with st.expander("🔍 Data-quality checks",
                 expanded=bool(pd.notna(ident) and ident > 0.5)):
    c1, c2, c3 = st.columns(3)
    c1.metric("Excluded rows where ML = user FC",
              "—" if pd.isna(ident) else f"{ident:0.0%}",
              help="If the ML column merely echoes the user forecast on "
                   "excluded items, the engine's shadow forecast was not "
                   "captured and the comparison is blind exactly where it "
                   "matters.")
    c2.metric("Rows with zero shipments", f"{(df['act'] == 0).mean():0.0%}")
    c3.metric("Periods analysed", f"{len(months_sorted)}")
    if pd.notna(ident) and ident > 0.5:
        st.error("More than half of the non-zero excluded rows have an "
                 "identical ML and user forecast — the ML column likely holds "
                 "the *adopted* forecast, not the engine's shadow forecast. "
                 "Fix the extract before acting on verdicts.")

# -----------------------------------------------------------------------------
# 6 · Scorecard (All / Y / N)
# -----------------------------------------------------------------------------
st.subheader("Whole-dataset scorecard")
rows = []
for label, frame in [("All items", df),
                     ("Excluded (Y)", df[df["_excl"] == "Y"]),
                     ("Not excluded (N)", df[df["_excl"] == "N"])]:
    rows.append({"Scope": label, **simple_metrics(frame)})
sc = add_verdict(pd.DataFrame(rows), fca_threshold)
st.dataframe(sc, width="stretch", hide_index=True,
             column_config=COLUMN_CONFIG)

y = sc[sc["Scope"] == "Excluded (Y)"].iloc[0]
if pd.isna(y["FCA User − ML"]):
    st.info("No shipped volume on excluded items — no overall verdict.")
elif y["FCA User − ML"] >= fca_threshold:
    st.success(f"**Excluded items overall: 👤 the user forecast adds value** — "
               f"FCA {y['FCA User']:.0%} vs {y['FCA ML']:.0%} for ML.")
elif y["FCA User − ML"] <= -fca_threshold:
    st.error(f"**Excluded items overall: 🤖 ML would be more accurate** — "
             f"FCA {y['FCA ML']:.0%} vs {y['FCA User']:.0%} for the user "
             "forecast. Review the exclusion list.")
else:
    st.warning(f"**Excluded items overall: ≈ tie** — FCA difference "
               f"({y['FCA User − ML']:+.1%}) is below the "
               f"{fca_threshold:.0%} threshold.")

# -----------------------------------------------------------------------------
# 7 · Summed-up analysis by any column
# -----------------------------------------------------------------------------
st.subheader("Summed-up analysis by…")
cset1, cset2 = st.columns([1.6, 2.4])
with cset1:
    measure_cols = {col_month, "ml", "user", "act", col_ml, col_user, col_act}
    bd_options = [c for c in df.columns
                  if c not in measure_cols and c != "_excl"]
    bd_options = [col_excl] + [c for c in bd_options if c != col_excl]
    bcol = st.selectbox("Break down by", bd_options, index=0)
    excl_scope = st.radio("ML exclusion scope",
                          ["Both (split Y/N)", "Y only", "N only"],
                          horizontal=True)
    view_mode = st.radio("View", ["Per month", "Aggregated"], horizontal=True)
with cset2:
    _pp = parse_periods(months_sorted)
    if _pp is not None:
        _ymap = {v: (d.year if pd.notna(d) else None)
                 for v, d in zip(months_sorted, _pp)}
        _years = sorted({yv for yv in _ymap.values() if yv is not None})
        sel_years = st.multiselect("Years", _years, default=_years)
        period_opts = [m for m in months_sorted
                       if _ymap.get(m) in set(sel_years)]
    else:
        period_opts = months_sorted
    sel_periods = st.multiselect("Months / periods", period_opts,
                                 default=period_opts)

if not sel_periods:
    st.info("Select at least one period to see the summed-up analysis.")
    st.stop()

dfb = df[df[col_month].isin(sel_periods)]
if excl_scope == "Y only":
    dfb = dfb[dfb["_excl"] == "Y"]
elif excl_scope == "N only":
    dfb = dfb[dfb["_excl"] == "N"]
if dfb.empty:
    st.warning("No rows match the selected periods and exclusion scope.")
    st.stop()

split_by_excl = excl_scope.startswith("Both") and bcol != col_excl
keys = [bcol, "_excl"] if split_by_excl else [bcol]
group_keys = keys + ([col_month] if view_mode == "Per month" else [])

bsum = (dfb.groupby(group_keys, dropna=False)
        .agg(ml=("ml", "sum"), user=("user", "sum"), act=("act", "sum"))
        .reset_index())
metrics = pd.DataFrame(
    [simple_metrics(bsum.iloc[[i]]) for i in range(len(bsum))])
bsum = pd.concat([bsum[group_keys].reset_index(drop=True), metrics], axis=1)
bsum = add_verdict(bsum, fca_threshold)

# Order: biggest groups first; Y before N; months in order
vol_order = (bsum.groupby(bcol)["Shipped"].sum()
             .sort_values(ascending=False).index.tolist())
bsum["_r"] = bsum[bcol].map({v: i for i, v in enumerate(vol_order)})
sort_keys, sort_asc = ["_r"], [True]
if split_by_excl:
    sort_keys.append("_excl"); sort_asc.append(False)
if view_mode == "Per month":
    sort_keys.append(col_month); sort_asc.append(True)
bsum = bsum.sort_values(sort_keys, ascending=sort_asc).drop(columns="_r")

st.caption(f"Scope: **{excl_scope}** · View: **{view_mode}** · "
           f"{len(sel_periods)} period(s)")
disp = bsum.rename(columns={"_excl": "Excluded"})
show_cols = ([str(bcol)]
             + (["Excluded"] if split_by_excl else [])
             + ([str(col_month)] if view_mode == "Per month" else [])
             + NUM_COLS + PCT_COLS + ["Verdict"])
st.dataframe(disp[show_cols], width="stretch", hide_index=True,
             column_config=COLUMN_CONFIG)
st.download_button(
    "⬇️ Download this table (CSV)",
    disp[show_cols].to_csv(index=False).encode(),
    "summed_up_analysis.csv", "text/csv")

# ---- FCA chart (always aggregated view) --------------------------------------
csum = (dfb.groupby(keys, dropna=False)
        .agg(ml=("ml", "sum"), user=("user", "sum"), act=("act", "sum"))
        .reset_index())
cmet = pd.DataFrame(
    [simple_metrics(csum.iloc[[i]]) for i in range(len(csum))])
csum = pd.concat([csum[keys].reset_index(drop=True), cmet], axis=1)
csum = csum.sort_values("Shipped", ascending=False)
if view_mode == "Per month":
    st.caption("Chart shows the selected periods aggregated.")

if split_by_excl:
    top_groups = (csum.groupby(bcol)["Shipped"].sum()
                  .sort_values(ascending=False).head(12).index.tolist())
    plot_bd = csum[csum[bcol].isin(top_groups) & csum["FCA ML"].notna()]
    if not plot_bd.empty:
        long = plot_bd.melt(id_vars=[bcol, "_excl"],
                            value_vars=["FCA ML", "FCA User"],
                            var_name="side", value_name="fca")
        figb = px.bar(long, x=bcol, y="fca", color="side", barmode="group",
                      facet_col="_excl",
                      color_discrete_map={"FCA ML": "#ef4444",
                                          "FCA User": "#3b82f6"},
                      category_orders={bcol: [str(g) for g in top_groups],
                                       "_excl": ["Y", "N"]},
                      labels={"fca": "FCA", "side": ""})
        figb.for_each_annotation(lambda a: a.update(
            text="Excluded (Y)" if a.text.endswith("Y")
            else ("Not excluded (N)" if a.text.endswith("N") else a.text)))
        figb.update_layout(height=440, legend=dict(orientation="h", y=1.12))
        figb.update_yaxes(tickformat=".0%")
        st.plotly_chart(figb, width="stretch")
        st.caption("Top 12 groups by shipped volume.")
else:
    plot_bd = csum[csum["FCA ML"].notna()].head(20)
    if len(plot_bd) > 1:
        figb = go.Figure()
        figb.add_bar(x=as_str(plot_bd[bcol]), y=plot_bd["FCA ML"],
                     name="FCA ML", marker_color="#ef4444")
        figb.add_bar(x=as_str(plot_bd[bcol]), y=plot_bd["FCA User"],
                     name="FCA User", marker_color="#3b82f6")
        figb.update_layout(barmode="group", height=420,
                           yaxis_tickformat=".0%", yaxis_title="FCA",
                           xaxis_title=str(bcol),
                           legend=dict(orientation="h", y=1.1))
        st.plotly_chart(figb, width="stretch")
        st.caption("Top 20 groups by shipped volume.")

st.caption("Methodology: Variance = FC − Shipped (signed units) · "
           "Bias % = Variance / Shipped · FCA = 1 − |Variance| / Shipped, "
           "floored at 0. Computed on sums at the selected grouping — over- "
           "and under-forecasts within a group net out, which is intended in "
           "this simple view.")
