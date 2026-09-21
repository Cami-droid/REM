"""
app.py - Streamlit: evolucion de las proyecciones del REM (BCRA) y desvio respecto de los resultados reales.
Lee data/rem_long.csv y data/rem_errors.csv (build_data.py / compute_errors.py) y, opcional, data/hitos.csv
(columnas: fecha, etiqueta) para marcar hitos en los graficos temporales.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.colors import sample_colorscale

DATA = Path(__file__).parent / "data"
KEY = ["relevamiento", "variable", "unidad_norm", "periodo_tipo", "fecha_objetivo"]
NIVELES = {"TC_NOMINAL", "EXPORTACIONES", "IMPORTACIONES"}
APROX_TASA = {"TASA_LEBAC35", "TASA_PASE7", "TASA_LELIQ"}
TIPO_LABEL = {"mes": "Mensual", "trim": "Trimestral", "anio": "Anual (cierre/promedio del año)",
              "prox_12m": "Próximos 12 meses", "prox_24m": "Próximos 24 meses (12 a 24)"}
# rezago de publicacion (meses) para definir "ultimo dato conocido" al momento del relevamiento (benchmark ingenuo)
LAGS = {"mes": 0, "trim": 2, "anio": 2, "prox_12m": 0, "prox_24m": 0}
st.set_page_config(page_title="REM: proyecciones vs. realidad", layout="wide")


# ---------------------------------------------------------------- datos
@st.cache_data(show_spinner=False)
def load():
    long = pd.read_csv(DATA / "rem_long.csv", parse_dates=["fecha_objetivo"])
    err = pd.read_csv(DATA / "rem_errors.csv", parse_dates=["fecha_objetivo"])
    long["unidad_norm"] = long["unidad"].str.replace("US$", "USD", regex=False)  # igual que compute_errors.py
    long.loc[long["variable"].str.startswith("TASA_"), "unidad_norm"] = "TNA; %"
    long = long.merge(err[KEY + ["real", "comparable"]].drop_duplicates(KEY), on=KEY, how="left")
    long["comparable"] = long["comparable"].fillna(True).astype(bool)
    for d in (long, err):
        d["rel_dt"] = pd.to_datetime(d["relevamiento"] + "-01")
        d["rel_year"] = d["rel_dt"].dt.year
    return long, err


@st.cache_data(show_spinner=False)
def load_hitos():
    p = DATA / "hitos.csv"
    if not p.exists():
        return pd.DataFrame(columns=["fecha", "etiqueta"])
    h = pd.read_csv(p, parse_dates=["fecha"])
    return h.dropna(subset=["fecha"])


def with_naive(e_, tipo):
    """Agrega el pronostico ingenuo (ultimo real conocido al relevamiento) y su error. Sin fuga: el dato conocido
    debe ser anterior al periodo objetivo."""
    lag = LAGS.get(tipo, 0)
    parts = []
    for sr, g in e_.groupby("serie_real"):
        rs = (ERR.loc[(ERR["serie_real"] == sr) & ERR["real"].notna(), ["fecha_objetivo", "real"]]
              .drop_duplicates("fecha_objetivo").sort_values("fecha_objetivo")
              .rename(columns={"fecha_objetivo": "f_known", "real": "naive"}))
        rs["f_known"] = rs["f_known"].astype("datetime64[ns]")
        g = g.copy()
        g["cutoff"] = (g["rel_dt"] - pd.DateOffset(months=lag)).astype("datetime64[ns]")
        m = pd.merge_asof(g.sort_values("cutoff"), rs, left_on="cutoff", right_on="f_known", direction="backward")
        m.loc[m["f_known"] >= m["fecha_objetivo"], "naive"] = np.nan
        parts.append(m)
    out = pd.concat(parts) if parts else e_.assign(naive=np.nan)
    out["naive_err"] = out["real"] - out["naive"]
    out["naive_abs"] = out["naive_err"].abs()
    return out


def metrics(x):
    """sesgo, MAE, cobertura y ratio MAE REM / MAE ingenuo (<1: el REM le gana al ingenuo)."""
    n = len(x)
    ok = x["naive_abs"].notna()
    mae_n = x.loc[ok, "naive_abs"].mean() if ok.any() else np.nan
    mae_r = x.loc[ok, "abs_error"].mean() if ok.any() else np.nan
    return dict(n=n, sesgo=x["error"].mean(), MAE=x["abs_error"].mean(),
                cobertura=x["dentro_p10_p90"].mean(), MAE_ingenuo=mae_n,
                ratio=(mae_r / mae_n) if mae_n and mae_n > 0 else np.nan)


@st.cache_data(show_spinner=False)
def summary(solo_comp, h0, h1):
    x = ERR[ERR["comparable"]] if solo_comp else ERR
    x = x[x["horizonte_meses"].between(h0, h1)]
    rows = []
    for (v, u, t), g in x.groupby(["variable", "unidad_norm", "periodo_tipo"]):
        m = metrics(with_naive(g, t))
        rows.append(dict(variable=v, unidad=u, periodo=TIPO_LABEL.get(t, t), n=m["n"], sesgo=m["sesgo"],
                         MAE=m["MAE"], cobertura=m["cobertura"], ratio_vs_ingenuo=m["ratio"]))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- helpers de graficos
def fmt_obj(ts, tipo):
    ts = pd.Timestamp(ts)
    if tipo == "anio":
        return f"{ts.year}"
    if tipo == "trim":
        return f"{ts.year}-T{(ts.month - 1) // 3 + 1}"
    return f"{ts:%Y-%m}"


def xf(s, tipo):  # anual: eje en anios enteros (no en 31-dic)
    return s.dt.year if tipo == "anio" else s


def add_hitos(fig, annual=False):
    if not st.session_state.get("show_hitos", True):
        return fig
    for _, r in load_hitos().iterrows():
        x = int(r["fecha"].year) if annual else r["fecha"].isoformat()
        fig.add_shape(type="line", x0=x, x1=x, y0=0, y1=1, yref="paper",
                      line=dict(color="rgba(150,150,150,0.6)", dash="dot", width=1))
        fig.add_annotation(x=x, y=1, yref="paper", text=str(r.get("etiqueta", "")), showarrow=False,
                           textangle=-90, xanchor="left", yanchor="top", font=dict(size=10, color="gray"))
    return fig


def unit_label(unidad):
    return "pp" if "%" in unidad else unidad


def frase(unidad, n, h0, h1, m):
    u = unit_label(unidad)
    hor = f"A {h0} meses de horizonte" if h0 == h1 else f"Entre {h0} y {h1} meses de horizonte"
    verbo = "subestimó" if m["sesgo"] > 0 else "sobreestimó"
    s = (f"{hor} (n = {n}), el REM {verbo} en promedio {abs(m['sesgo']):,.2f} {u}; "
         f"el error absoluto medio fue {m['MAE']:,.2f} {u}.")
    if m["cobertura"] == m["cobertura"]:
        s += f" El real cayó dentro de p10–p90 en el {m['cobertura']:.0%} de los casos (nominal: 80%)."
    if m["ratio"] == m["ratio"]:
        s += (f" Su MAE es {m['ratio']:.2f}× el de repetir el último dato conocido: "
              f"{'mejor' if m['ratio'] < 1 else 'peor'} que el pronóstico ingenuo.")
    return s


# ---------------------------------------------------------------- carga
try:
    LONG, ERR = load()
except FileNotFoundError as ex:
    st.error(f"Falta {ex.filename}. Correr build_data.py / compute_errors.py y commitear data/.")
    st.stop()

# ---------------------------------------------------------------- sidebar (con estado en la URL)
qp = st.query_params
st.sidebar.header("Filtros")
variables = sorted(LONG["variable"].unique())
if "variable" not in st.session_state and qp.get("v") in variables:
    st.session_state["variable"] = qp["v"]
variable = st.sidebar.selectbox("Variable", variables, key="variable")
sv = LONG[LONG["variable"] == variable]
unidades = sorted(sv["unidad_norm"].unique())
if f"unidad|{variable}" not in st.session_state and qp.get("u") in unidades:
    st.session_state[f"unidad|{variable}"] = qp["u"]
unidad = st.sidebar.selectbox("Unidad", unidades, key=f"unidad|{variable}")
sv = sv[sv["unidad_norm"] == unidad]
tipos = [t for t in TIPO_LABEL if t in set(sv["periodo_tipo"])]
if f"tipo|{variable}|{unidad}" not in st.session_state and qp.get("t") in tipos:
    st.session_state[f"tipo|{variable}|{unidad}"] = qp["t"]
tipo = st.sidebar.selectbox("Tipo de período objetivo", tipos, format_func=TIPO_LABEL.get,
                            key=f"tipo|{variable}|{unidad}")
solo_comp = st.sidebar.checkbox("Solo períodos comparables", value=True,
                                help="Excluye IPC nacional anterior a 2017 (INDEC no publicaba nivel nacional).")
st.session_state["show_hitos"] = st.sidebar.checkbox("Mostrar hitos (data/hitos.csv)", value=True)
st.query_params.update(v=variable, u=unidad, t=tipo)

base = sv[sv["periodo_tipo"] == tipo]
if solo_comp:
    base = base[base["comparable"]]
rels = sorted(base["relevamiento"].unique())
if not rels:
    st.warning("Sin datos para esta combinación.")
    st.stop()
r0, r1 = (st.sidebar.select_slider("Relevamientos", options=rels, value=(rels[0], rels[-1]))
          if len(rels) > 1 else (rels[0], rels[0]))
view = base[(base["relevamiento"] >= r0) & (base["relevamiento"] <= r1)]

e0 = ERR[(ERR["variable"] == variable) & (ERR["unidad_norm"] == unidad) & (ERR["periodo_tipo"] == tipo)]
if solo_comp:
    e0 = e0[e0["comparable"]]
e0 = e0[(e0["relevamiento"] >= r0) & (e0["relevamiento"] <= r1)]
ctx = f"{variable}|{unidad}|{tipo}|{solo_comp}|{r0}|{r1}"

if e0.empty:
    ee, h0, h1 = e0, 0, 0
else:
    hmin, hmax = int(e0["horizonte_meses"].min()), int(e0["horizonte_meses"].max())
    if hmin < hmax:
        h0, h1 = st.sidebar.slider("Horizonte (meses) para errores", hmin, hmax, (max(hmin, 0) if max(hmin, 0) < hmax else hmin, hmax),
                                   key=f"hz|{ctx}")
    else:
        h0, h1 = hmin, hmax
    ee = e0[e0["horizonte_meses"].between(h0, h1)]
een = with_naive(ee, tipo) if not ee.empty else ee.assign(naive=np.nan, naive_err=np.nan, naive_abs=np.nan)

# ---------------------------------------------------------------- encabezado
st.title(f"REM: {variable} · {unidad}")
if variable in APROX_TASA:
    st.caption("Ojo: el BCRA publica una sola serie de tasa de política (empalmada); es una aproximación para esta variable.")

with st.expander("Cómo leer esto"):
    st.markdown("""
- **REM**: Relevamiento de Expectativas de Mercado del BCRA. Cada *relevamiento* (mes) publica la **mediana** de las proyecciones
  para distintos períodos objetivo y el rango **p10–p90** (entre el 10% y el 90% de los analistas: 80% nominal).
- **Horizonte**: meses entre el relevamiento y el período objetivo. **Error = real − mediana**: positivo = el REM *subestimó*;
  negativo = *sobreestimó*. Las unidades son las de la variable (pp, $, millones de USD…).
- **Sesgo**: error medio (¿se equivoca siempre para el mismo lado?). **MAE**: error absoluto medio (¿cuánto se equivoca?).
  **Cobertura p10–p90**: % de veces que el real cayó dentro del rango; si es muy inferior al 80%, el REM subestima la incertidumbre.
- **Pronóstico ingenuo**: repetir el último dato real conocido al momento del relevamiento (con rezago de publicación:
  0 meses en series mensuales, 2 en trimestrales y anuales). **Ratio < 1**: el REM le gana al ingenuo.
- **Reales**: tipo de cambio y tasas = promedio mensual; IPC y demás = dato del período. LEBAC/Pase/LELIQ se comparan contra una única
  serie de tasa de política (aproximación). El PIB se revisa: se usa la última versión publicada.
- **Comparable**: se excluye el IPC nacional anterior a 2017 (no había nivel nacional del INDEC).
- **Hitos**: líneas punteadas desde `data/hitos.csv` (`fecha,etiqueta`), editable. La URL guarda variable/unidad/tipo para compartir.
""")

# tarjetas resumen + frase
if een.empty:
    st.info("Todavía no hay resultados reales comparables para esta selección; se muestran solo las proyecciones.")
else:
    m = metrics(een)
    u = unit_label(unidad)
    c = st.columns(5)
    c[0].metric("Sesgo (real − mediana)", f"{m['sesgo']:+,.2f} {u}")
    c[1].metric("MAE", f"{m['MAE']:,.2f} {u}")
    c[2].metric("Cobertura p10–p90", f"{m['cobertura']:.0%}" if m["cobertura"] == m["cobertura"] else "s/d",
                delta="nominal 80%", delta_color="off")
    c[3].metric("Observaciones (n)", f"{m['n']:,}")
    c[4].metric("MAE REM / MAE ingenuo", f"{m['ratio']:.2f}" if m["ratio"] == m["ratio"] else "s/d",
                delta=("REM mejor" if m["ratio"] < 1 else "REM peor") if m["ratio"] == m["ratio"] else None,
                delta_color="off")
    st.info(frase(unidad, m["n"], h0, h1, m))

t_res, t_evo, t_hor, t_tie = st.tabs(["Resumen general", "Evolución de proyecciones", "Error por horizonte",
                                       "Errores en el tiempo"])

# ---------------------------------------------------------------- resumen general (portada)
with t_res:
    st.caption("Todas las variables en una tabla (respeta «Solo períodos comparables»). Ratio < 1: el REM le gana al pronóstico ingenuo.")
    ph = st.slider("Horizonte (meses)", 0, 24, (0, 12), key="ph_res")
    sm = summary(solo_comp, *ph)
    if sm.empty:
        st.info("Sin datos.")
    else:
        sm["cobertura"] = sm["cobertura"] * 100
        st.dataframe(sm.round(3), width="stretch", hide_index=True, column_config={
            "cobertura": st.column_config.ProgressColumn("cobertura p10–p90", min_value=0.0, max_value=100.0,
                                                         format="%.0f%%"),
            "ratio_vs_ingenuo": st.column_config.NumberColumn("MAE REM / ingenuo", format="%.2f"),
        })
        st.caption("El sesgo y el MAE están en las unidades de cada variable: no son comparables entre filas; "
                   "para comparar usar cobertura y ratio vs. ingenuo.")

# ---------------------------------------------------------------- evolucion
with t_evo:
    modos = ["Por período objetivo", "Horizonte fijo vs. real"]
    if tipo in ("mes", "trim", "anio"):
        modos = ["Por período objetivo", "Trayectorias por relevamiento", "Horizonte fijo vs. real",
                 "Último relevamiento (fan chart)"]
    modo = st.radio("Vista", modos, horizontal=True)
    fig = go.Figure()
    band = lambda f, x, lo, hi: (f.add_trace(go.Scatter(x=x, y=hi, mode="lines", line=dict(width=0),
                                                        showlegend=False, hoverinfo="skip")),
                                 f.add_trace(go.Scatter(x=x, y=lo, mode="lines", line=dict(width=0), fill="tonexty",
                                                        fillcolor="rgba(31,119,180,0.2)", name="p10–p90")))
    realr = base.dropna(subset=["real"]).drop_duplicates("fecha_objetivo").sort_values("fecha_objetivo")
    annual_axis = False

    if modo == "Por período objetivo":
        objs = sorted(view["fecha_objetivo"].unique(), reverse=True)
        con_real = sorted(view.loc[view["real"].notna(), "fecha_objetivo"].unique(), reverse=True)
        obj = st.selectbox("Período objetivo", objs, index=objs.index(con_real[0]) if con_real else 0,
                           format_func=lambda t: fmt_obj(t, tipo), key=f"obj|{ctx}")
        d = view[view["fecha_objetivo"] == obj].sort_values("rel_dt")
        if d["p10"].notna().any():
            band(fig, d["rel_dt"], d["p10"], d["p90"])
        cd = np.stack([d["p10"], d["p90"], d["real"], d["real"] - d["mediana"]], axis=-1)
        fig.add_trace(go.Scatter(x=d["rel_dt"], y=d["mediana"], mode="lines+markers", name="Mediana REM",
                                 line=dict(color="#1f77b4"), customdata=cd,
                                 hovertemplate="Relevamiento %{x|%Y-%m}<br>Mediana: %{y:,.2f}<br>p10–p90: "
                                               "%{customdata[0]:,.2f} – %{customdata[1]:,.2f}<br>Real: "
                                               "%{customdata[2]:,.2f}<br>Error: %{customdata[3]:+,.2f}<extra></extra>"))
        real = d["real"].dropna()
        if not real.empty:
            fig.add_hline(y=real.iloc[0], line=dict(color="black", dash="dash"),
                          annotation_text=f"Real: {real.iloc[0]:,.2f}", annotation_position="top left")
        fig.update_layout(title=f"Qué proyectaba cada relevamiento para {fmt_obj(obj, tipo)}",
                          xaxis_title="Relevamiento", yaxis_title=unidad)

    elif modo == "Trayectorias por relevamiento":
        opts = sorted(view["relevamiento"].unique())
        AUTO = {"2 por año": 2, "1 por año": 1, "4 por año": 4, "Manual": 0}
        auto = st.radio("Selección automática de relevamientos", list(AUTO), horizontal=True,
                        help="Elige relevamientos representativos de cada año (el más cercano a jun/dic, etc.) "
                             "y el último disponible. Después se puede ajustar a mano.")
        k = AUTO[auto]
        if k:
            targets = {1: [12], 2: [6, 12], 4: [3, 6, 9, 12]}[k]
            by_year = {}
            for r in opts:
                by_year.setdefault(int(r[:4]), []).append((int(r[5:7]), r))
            default = sorted({min(l, key=lambda x: abs(x[0] - t))[1] for l in by_year.values() for t in targets}
                             | {opts[-1]})
        else:
            default = sorted({opts[i] for i in np.linspace(0, len(opts) - 1, min(5, len(opts))).astype(int)})
        sel = st.multiselect("Relevamientos a superponer", opts, default=default, key=f"sel|{ctx}|{auto}")
        if len(sel) > 20:
            st.warning(f"{len(sel)} series superpuestas: el gráfico puede volverse ilegible. Probá con menos.")
        annual_axis = tipo == "anio"
        if not realr.empty:
            fig.add_trace(go.Scatter(x=xf(realr["fecha_objetivo"], tipo), y=realr["real"], mode="lines+markers",
                                     name="Real", line=dict(color="black", width=3)))
        cols = sample_colorscale("Viridis", [i / max(len(sel) - 1, 1) for i in range(len(sel))])
        for r, c_ in zip(sorted(sel), cols):  # violeta (viejos) -> amarillo (recientes)
            d = view[view["relevamiento"] == r].sort_values("fecha_objetivo")
            fig.add_trace(go.Scatter(x=xf(d["fecha_objetivo"], tipo), y=d["mediana"], mode="lines+markers",
                                     name=f"REM {r}", line=dict(width=1.8, color=c_), marker=dict(size=5)))
        fig.update_layout(title="Trayectorias proyectadas (mediana) vs. real", xaxis_title="Período objetivo",
                          yaxis_title=unidad)
        if annual_axis:
            fig.update_xaxes(dtick=1)

    elif modo == "Horizonte fijo vs. real":
        hs = sorted(view["horizonte_meses"].dropna().unique())
        hs = [int(h) for h in hs]
        hsel = st.selectbox("Horizonte (meses antes del período objetivo)", hs,
                            index=hs.index(12) if 12 in hs else len(hs) // 2, key=f"hfix|{ctx}")
        d = view[view["horizonte_meses"] == hsel].sort_values("fecha_objetivo")
        annual_axis = tipo == "anio"
        x = xf(d["fecha_objetivo"], tipo)
        if d["p10"].notna().any():
            band(fig, x, d["p10"], d["p90"])
        cd = np.stack([d["relevamiento"], d["real"], d["real"] - d["mediana"]], axis=-1)
        fig.add_trace(go.Scatter(x=x, y=d["mediana"], mode="lines+markers", name=f"REM a {hsel} meses",
                                 line=dict(color="#1f77b4"), customdata=cd,
                                 hovertemplate="Relevamiento %{customdata[0]}<br>Mediana: %{y:,.2f}<br>Real: "
                                               "%{customdata[1]:,.2f}<br>Error: %{customdata[2]:+,.2f}<extra></extra>"))
        if not realr.empty:
            fig.add_trace(go.Scatter(x=xf(realr["fecha_objetivo"], tipo), y=realr["real"], mode="lines+markers",
                                     name="Real", line=dict(color="black", width=2.5)))
        fig.update_layout(title=f"Lo que el REM proyectaba {hsel} meses antes vs. lo que pasó",
                          xaxis_title="Período objetivo", yaxis_title=unidad)
        if annual_axis:
            fig.update_xaxes(dtick=1)

    else:  # fan chart del ultimo relevamiento
        last = view["relevamiento"].max()
        d = view[view["relevamiento"] == last].sort_values("fecha_objetivo")
        annual_axis = tipo == "anio"
        x = xf(d["fecha_objetivo"], tipo)
        if not realr.empty:
            fig.add_trace(go.Scatter(x=xf(realr["fecha_objetivo"], tipo), y=realr["real"], mode="lines",
                                     name="Real (histórico)", line=dict(color="black", width=2.5)))
        if d["p10"].notna().any():
            band(fig, x, d["p10"], d["p90"])
        fig.add_trace(go.Scatter(x=x, y=d["mediana"], mode="lines+markers", name=f"Mediana REM {last}",
                                 line=dict(color="#d62728")))
        fig.update_layout(title=f"Proyecciones del último relevamiento ({last}) y real histórico",
                          xaxis_title="Período objetivo", yaxis_title=unidad)
        if annual_axis:
            fig.update_xaxes(dtick=1)

    if modo != "Por período objetivo":
        add_hitos(fig, annual=annual_axis)
    else:
        add_hitos(fig, annual=False)
    fig.update_layout(hovermode="x unified" if modo != "Por período objetivo" else "closest",
                      legend=dict(orientation="h", y=-0.2), height=520)
    st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------- error por horizonte
with t_hor:
    st.caption("error = real − mediana (>0: el REM subestimó; <0: sobreestimó). Cobertura nominal p10–p90 = 80%. "
               "El rango de horizontes se elige en la barra lateral.")
    if een.empty:
        st.info("Sin errores calculables.")
    else:
        nmin = st.slider("n mínimo por horizonte", 1, 30, 3)
        g = een.groupby("horizonte_meses").apply(
            lambda x: pd.Series(metrics(x)), include_groups=False)
        g["RMSE"] = een.groupby("horizonte_meses")["error"].apply(lambda x: float(np.sqrt((x ** 2).mean())))
        g["error_rel_abs_pct"] = een.groupby("horizonte_meses")["error_rel_pct"].apply(lambda x: x.abs().mean())
        g = g[g["n"] >= nmin].reset_index()
        if g.empty:
            st.info("Ningún horizonte alcanza el n mínimo.")
        else:
            f1 = go.Figure()
            f1.add_trace(go.Bar(x=g["horizonte_meses"], y=g["sesgo"], name="Sesgo (error medio)", marker_color="#ff7f0e"))
            f1.add_trace(go.Scatter(x=g["horizonte_meses"], y=g["MAE"], name="MAE REM", mode="lines+markers",
                                    line=dict(color="#1f77b4")))
            f1.add_trace(go.Scatter(x=g["horizonte_meses"], y=g["MAE_ingenuo"], name="MAE ingenuo (último dato)",
                                    mode="lines+markers", line=dict(color="gray", dash="dash")))
            f1.add_hline(y=0, line=dict(color="gray", width=1))
            f1.update_layout(title="Sesgo, MAE del REM y MAE del pronóstico ingenuo por horizonte",
                             xaxis_title="Horizonte (meses)", yaxis_title=unidad, height=420,
                             legend=dict(orientation="h", y=-0.25))
            st.plotly_chart(f1, width="stretch")
            st.caption("El MAE ingenuo se calcula sobre las mismas observaciones en las que existe un último dato conocido.")

            if g["cobertura"].notna().any():
                f2 = go.Figure(go.Scatter(x=g["horizonte_meses"], y=g["cobertura"], mode="lines+markers"))
                f2.add_hline(y=0.8, line=dict(color="gray", dash="dash"), annotation_text="80% nominal")
                f2.update_layout(title="Cobertura p10–p90 por horizonte", xaxis_title="Horizonte (meses)",
                                 yaxis=dict(range=[0, 1], tickformat=".0%"), height=330)
                st.plotly_chart(f2, width="stretch")

            hh = een[een["horizonte_meses"].isin(g["horizonte_meses"])]
            f3 = go.Figure(go.Box(x=hh["horizonte_meses"], y=hh["error"], boxpoints=False, marker_color="#1f77b4",
                                  name="error"))
            f3.add_hline(y=0, line=dict(color="gray", width=1))
            f3.update_layout(title="Distribución del error por horizonte (caja: p25–p75; bigotes: rango sin atípicos)",
                             xaxis_title="Horizonte (meses)", yaxis_title=f"error ({unidad})", height=380,
                             showlegend=False)
            st.plotly_chart(f3, width="stretch")

            met = st.radio("Heatmap: año del relevamiento × horizonte", ["Sesgo", "MAE"], horizontal=True)
            colv = "error" if met == "Sesgo" else "abs_error"
            pv = hh.pivot_table(index="rel_year", columns="horizonte_meses", values=colv, aggfunc="mean")
            f4 = go.Figure(go.Heatmap(z=pv.values, x=pv.columns, y=[str(i) for i in pv.index],
                                      colorscale="RdBu_r" if met == "Sesgo" else "YlOrRd",
                                      zmid=0 if met == "Sesgo" else None, colorbar=dict(title=unit_label(unidad)),
                                      hovertemplate="Año relev. %{y}<br>Horizonte %{x}m<br>" + met + ": %{z:,.2f}<extra></extra>"))
            f4.update_layout(title=f"{met} por año de relevamiento y horizonte", xaxis_title="Horizonte (meses)",
                             yaxis=dict(type="category", autorange="reversed"), height=420)
            st.plotly_chart(f4, width="stretch")
            if met == "Sesgo":
                st.caption("Rojo: el REM subestimó (real > mediana); azul: sobreestimó.")

            if variable in NIVELES:
                st.caption("Para niveles, `error_rel_abs_pct` = |error| / real, promedio (%).")
            else:
                g = g.drop(columns="error_rel_abs_pct")
            st.dataframe(g.round(3), width="stretch", hide_index=True)
            st.download_button("Descargar tabla (CSV)", g.to_csv(index=False).encode("utf-8"),
                               file_name=f"rem_error_horizonte_{variable}.csv", mime="text/csv")

# ---------------------------------------------------------------- errores en el tiempo
with t_tie:
    if een.empty:
        st.info("Sin errores calculables.")
    else:
        annual_axis = tipo == "anio"
        f5 = go.Figure(go.Scatter(
            x=xf(een["fecha_objetivo"], tipo), y=een["error"], mode="markers",
            marker=dict(size=6, color=een["horizonte_meses"], colorscale="Viridis", showscale=True,
                        colorbar=dict(title="Horizonte<br>(meses)")),
            customdata=np.stack([een["relevamiento"], een["horizonte_meses"], een["mediana"], een["real"]], axis=-1),
            hovertemplate="Objetivo %{x}<br>Relevamiento %{customdata[0]} (h=%{customdata[1]}m)<br>Mediana: "
                          "%{customdata[2]:,.2f}<br>Real: %{customdata[3]:,.2f}<br>Error: %{y:+,.2f}<extra></extra>"))
        f5.add_hline(y=0, line=dict(color="gray", width=1))
        f5.update_layout(title="Error (real − mediana) por período objetivo", xaxis_title="Período objetivo",
                         yaxis_title=f"error ({unidad})", height=450)
        if annual_axis:
            f5.update_xaxes(dtick=1)
        add_hitos(f5, annual=annual_axis)
        st.plotly_chart(f5, width="stretch")

        lo, hi = float(min(een["mediana"].min(), een["real"].min())), float(max(een["mediana"].max(), een["real"].max()))
        f6 = go.Figure()
        f6.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color="gray", dash="dash"),
                                name="Proyección perfecta", hoverinfo="skip"))
        f6.add_trace(go.Scatter(
            x=een["mediana"], y=een["real"], mode="markers", name="Observaciones",
            marker=dict(size=6, color=een["horizonte_meses"], colorscale="Viridis", showscale=True,
                        colorbar=dict(title="Horizonte<br>(meses)")),
            customdata=np.stack([een["relevamiento"], een["fecha_objetivo"].dt.strftime("%Y-%m"),
                                 een["horizonte_meses"]], axis=-1),
            hovertemplate="Proyectado: %{x:,.2f}<br>Real: %{y:,.2f}<br>Relevamiento %{customdata[0]} → "
                          "%{customdata[1]} (h=%{customdata[2]}m)<extra></extra>"))
        f6.update_layout(title="Proyectado (mediana) vs. real", xaxis_title=f"proyectado ({unidad})",
                         yaxis_title=f"real ({unidad})", height=450)
        st.plotly_chart(f6, width="stretch")
        st.caption("Sobre la diagonal: el REM subestimó; debajo: sobreestimó.")

        st.subheader("10 mayores errores absolutos")
        top = een.sort_values("abs_error", ascending=False).head(10).copy()
        top["período objetivo"] = top["fecha_objetivo"].map(lambda t: fmt_obj(t, tipo))
        cols_top = ["relevamiento", "período objetivo", "horizonte_meses", "mediana", "real", "error"]
        if variable in NIVELES:
            cols_top.append("error_rel_pct")
        st.dataframe(top[cols_top].round(2), width="stretch", hide_index=True)
