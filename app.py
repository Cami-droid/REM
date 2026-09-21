"""
app.py - Streamlit: evolucion de las proyecciones del REM (BCRA) y desvio respecto de los resultados reales.
Solo lee data/rem_long.csv y data/rem_errors.csv (generados por build_data.py / compute_errors.py).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DATA = Path(__file__).parent / "data"
KEY = ["relevamiento", "variable", "unidad_norm", "periodo_tipo", "fecha_objetivo"]
NIVELES = {"TC_NOMINAL", "EXPORTACIONES", "IMPORTACIONES"}
APROX_TASA = {"TASA_LEBAC35", "TASA_PASE7", "TASA_LELIQ"}
TIPO_LABEL = {"mes": "Mensual", "trim": "Trimestral", "anio": "Anual (cierre/promedio del año)",
              "prox_12m": "Próximos 12 meses", "prox_24m": "Próximos 24 meses (12 a 24)"}
st.set_page_config(page_title="REM: proyecciones vs. realidad", layout="wide")


@st.cache_data(show_spinner=False)
def load():
    long = pd.read_csv(DATA / "rem_long.csv", parse_dates=["fecha_objetivo"])
    err = pd.read_csv(DATA / "rem_errors.csv", parse_dates=["fecha_objetivo"])
    # misma normalizacion de unidad que compute_errors.py
    long["unidad_norm"] = long["unidad"].str.replace("US$", "USD", regex=False)
    long.loc[long["variable"].str.startswith("TASA_"), "unidad_norm"] = "TNA; %"
    long = long.merge(err[KEY + ["real", "comparable"]].drop_duplicates(KEY), on=KEY, how="left")
    long["comparable"] = long["comparable"].fillna(True).astype(bool)
    for d in (long, err):
        d["rel_dt"] = pd.to_datetime(d["relevamiento"] + "-01")
    return long, err


def fmt_obj(ts, tipo):
    ts = pd.Timestamp(ts)
    if tipo == "anio":
        return f"{ts.year}"
    if tipo == "trim":
        return f"{ts.year}-T{(ts.month - 1) // 3 + 1}"
    return f"{ts:%Y-%m}"


try:
    LONG, ERR = load()
except FileNotFoundError as e:
    st.error(f"Falta {e.filename}. Correr build_data.py / compute_errors.py y commitear data/.")
    st.stop()

# ---------------------------------------------------------------- sidebar
st.sidebar.header("Filtros")
variable = st.sidebar.selectbox("Variable", sorted(LONG["variable"].unique()), key="variable")
sv = LONG[LONG["variable"] == variable]
unidad = st.sidebar.selectbox("Unidad", sorted(sv["unidad_norm"].unique()), key=f"unidad|{variable}")
sv = sv[sv["unidad_norm"] == unidad]
tipos = [t for t in TIPO_LABEL if t in set(sv["periodo_tipo"])]
tipo = st.sidebar.selectbox("Tipo de período objetivo", tipos, format_func=TIPO_LABEL.get, key=f"tipo|{variable}|{unidad}")
solo_comp = st.sidebar.checkbox("Solo períodos comparables", value=True,
                                help="Excluye IPC nacional anterior a 2017 (INDEC no publicaba nivel nacional).")

base = sv[sv["periodo_tipo"] == tipo]
if solo_comp:
    base = base[base["comparable"]]
rels = sorted(base["relevamiento"].unique())
if not rels:
    st.warning("Sin datos para esta combinación.")
    st.stop()
r0, r1 = (st.sidebar.select_slider("Relevamientos", options=rels, value=(rels[0], rels[-1]))
          if len(rels) > 1 else (rels[0], rels[0]))
ctx = f"{variable}|{unidad}|{tipo}|{solo_comp}|{r0}|{r1}"
view = base[(base["relevamiento"] >= r0) & (base["relevamiento"] <= r1)]

e = ERR[(ERR["variable"] == variable) & (ERR["unidad_norm"] == unidad) & (ERR["periodo_tipo"] == tipo)]
if solo_comp:
    e = e[e["comparable"]]
e = e[(e["relevamiento"] >= r0) & (e["relevamiento"] <= r1)]

st.title(f"REM: {variable} · {unidad}")
if variable in APROX_TASA:
    st.caption("Ojo: el BCRA publica una sola serie de tasa de política (empalmada); es una aproximación para esta variable.")
if e.empty:
    st.info("Todavía no hay resultados reales comparables para esta selección; se muestran solo las proyecciones.")

tab1, tab2 = st.tabs(["Evolución de proyecciones", "Error por horizonte"])

# ---------------------------------------------------------------- tab 1
with tab1:
    modos = ["Por período objetivo"] + (["Trayectorias por relevamiento"] if tipo in ("mes", "trim", "anio") else [])
    modo = st.radio("Vista", modos, horizontal=True)
    fig = go.Figure()
    if modo == "Por período objetivo":
        objs = sorted(view["fecha_objetivo"].unique(), reverse=True)
        con_real = sorted(view.loc[view["real"].notna(), "fecha_objetivo"].unique(), reverse=True)
        obj = st.selectbox("Período objetivo", objs, index=objs.index(con_real[0]) if con_real else 0,
                           format_func=lambda t: fmt_obj(t, tipo), key=f"obj|{ctx}")
        d = view[view["fecha_objetivo"] == obj].sort_values("rel_dt")
        if d["p10"].notna().any():
            fig.add_trace(go.Scatter(x=d["rel_dt"], y=d["p90"], mode="lines", line=dict(width=0),
                                     showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=d["rel_dt"], y=d["p10"], mode="lines", line=dict(width=0), fill="tonexty",
                                     fillcolor="rgba(31,119,180,0.2)", name="p10–p90"))
        fig.add_trace(go.Scatter(x=d["rel_dt"], y=d["mediana"], mode="lines+markers", name="Mediana REM",
                                 line=dict(color="#1f77b4")))
        real = d["real"].dropna()
        if not real.empty:
            fig.add_hline(y=real.iloc[0], line=dict(color="black", dash="dash"),
                          annotation_text=f"Real: {real.iloc[0]:,.2f}", annotation_position="top left")
        fig.update_layout(title=f"Qué proyectaba cada relevamiento para {fmt_obj(obj, tipo)}",
                          xaxis_title="Relevamiento", yaxis_title=unidad)
    else:
        opts = sorted(view["relevamiento"].unique())
        default = sorted({opts[i] for i in np.linspace(0, len(opts) - 1, min(6, len(opts))).astype(int)})
        sel = st.multiselect("Relevamientos a superponer", opts, default=default, key=f"sel|{ctx}")
        realr = (base.dropna(subset=["real"]).drop_duplicates("fecha_objetivo").sort_values("fecha_objetivo"))
        if not realr.empty:
            fig.add_trace(go.Scatter(x=realr["fecha_objetivo"], y=realr["real"], mode="lines+markers", name="Real",
                                     line=dict(color="black", width=3)))
        for r in sel:
            d = view[view["relevamiento"] == r].sort_values("fecha_objetivo")
            fig.add_trace(go.Scatter(x=d["fecha_objetivo"], y=d["mediana"], mode="lines+markers",
                                     name=f"REM {r}", line=dict(width=1.5), marker=dict(size=4)))
        fig.update_layout(title="Trayectorias proyectadas (mediana) vs. real",
                          xaxis_title="Período objetivo", yaxis_title=unidad)
    fig.update_layout(hovermode="x unified", legend=dict(orientation="h", y=-0.2), height=520)
    st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------- tab 2
with tab2:
    st.caption("error = real − mediana (>0: el REM subestimó; <0: sobreestimó). Cobertura nominal p10–p90 = 80%.")
    if e.empty:
        st.info("Sin errores calculables.")
    else:
        hmin, hmax = int(e["horizonte_meses"].min()), int(e["horizonte_meses"].max())
        c1, c2 = st.columns(2)
        hr = c1.slider("Horizonte (meses)", hmin, hmax, (max(hmin, 0), hmax)) if hmin < hmax else (hmin, hmax)
        nmin = c2.slider("n mínimo por horizonte", 1, 30, 3)
        ee = e[e["horizonte_meses"].between(*hr)]
        g = ee.groupby("horizonte_meses").agg(
            n=("error", "size"), sesgo=("error", "mean"), MAE=("abs_error", "mean"),
            RMSE=("error", lambda x: float(np.sqrt((x ** 2).mean()))),
            cobertura_p10_p90=("dentro_p10_p90", "mean"), error_rel_abs_pct=("error_rel_pct", lambda x: x.abs().mean()))
        g = g[g["n"] >= nmin].reset_index()
        if g.empty:
            st.info("Ningún horizonte alcanza el n mínimo.")
        else:
            f1 = go.Figure()
            f1.add_trace(go.Bar(x=g["horizonte_meses"], y=g["sesgo"], name="Sesgo (error medio)",
                                marker_color="#ff7f0e"))
            f1.add_trace(go.Scatter(x=g["horizonte_meses"], y=g["MAE"], name="MAE", mode="lines+markers",
                                    line=dict(color="#1f77b4")))
            f1.update_layout(title="Sesgo y MAE por horizonte", xaxis_title="Horizonte (meses)",
                             yaxis_title=unidad, height=400, legend=dict(orientation="h", y=-0.25))
            f1.add_hline(y=0, line=dict(color="gray", width=1))
            st.plotly_chart(f1, width="stretch")
            if g["cobertura_p10_p90"].notna().any():
                f2 = go.Figure(go.Scatter(x=g["horizonte_meses"], y=g["cobertura_p10_p90"], mode="lines+markers"))
                f2.add_hline(y=0.8, line=dict(color="gray", dash="dash"), annotation_text="80% nominal")
                f2.update_layout(title="Cobertura p10–p90 por horizonte", xaxis_title="Horizonte (meses)",
                                 yaxis=dict(range=[0, 1], tickformat=".0%"), height=350)
                st.plotly_chart(f2, width="stretch")
            if variable in NIVELES:
                st.caption("Para niveles, `error_rel_abs_pct` = |error| / real, promedio (%).")
            else:
                g = g.drop(columns="error_rel_abs_pct")
            st.dataframe(g.round(3), width="stretch", hide_index=True)
            st.download_button("Descargar tabla (CSV)", g.to_csv(index=False).encode("utf-8"),
                               file_name=f"rem_error_horizonte_{variable}.csv", mime="text/csv")
