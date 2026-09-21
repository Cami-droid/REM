#!/usr/bin/env python
"""
compute_errors.py - Cruza las proyecciones del REM (rem_long.csv) con los resultados reales del BCRA
(actuals_monthly.csv) y genera data/rem_errors.csv, una fila por (relevamiento, variable, periodo objetivo).

Uso:
    python compute_errors.py [--data-dir data] [--out data/rem_errors.csv]

error = real - mediana   (>0: el REM subestimo; <0: sobreestimo). Unidades de la variable (pp, $, ...).
error_rel_pct solo para niveles (TC, exportaciones, importaciones): error / real * 100.
dentro_p10_p90: 1/0 si el real cayo dentro del rango p10-p90 de las proyecciones (NaN si no hay p10/p90).

Reales: BCRA (actuals_monthly.csv) e INDEC (actuals_indec.csv, opcional). Sin real todavia: resultado primario
del SPNF (no es INDEC: Secretaria de Hacienda).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# reglas: (variable REM, prefijo de 'unidad' o None, periodo_tipo o None, serie real, columna)
#   promedio = promedio mensual de la serie diaria | ultimo = dato del periodo (IPC, INDEC)
#   series BCRA: actuals_monthly.csv (fetch_actuals.py) | series INDEC: actuals_indec.csv (fetch_indec.py)
MI, IA = "var. % mensual", "var. % i.a."
RULES = [
    ("IPC_NG_NAC", MI, None, "IPC_MENSUAL", "ultimo"),
    ("IPC_NG_NAC", IA, None, "IPC_INTERANUAL", "ultimo"),
    ("IPC_NUCLEO_NAC", MI, None, "IPC_NUCLEO_MENSUAL", "ultimo"),
    ("IPC_NUCLEO_NAC", IA, None, "IPC_NUCLEO_INTERANUAL", "ultimo"),
    ("IPC_NG_GBA", MI, None, "IPC_GBA_MENSUAL", "ultimo"),
    ("IPC_NG_GBA", IA, None, "IPC_GBA_INTERANUAL", "ultimo"),
    ("IPC_NUCLEO_GBA", MI, None, "IPC_GBA_NUCLEO_MENSUAL", "ultimo"),
    ("IPC_NUCLEO_GBA", IA, None, "IPC_GBA_NUCLEO_INTERANUAL", "ultimo"),
    ("TC_NOMINAL", None, None, "TC_MAYORISTA", "promedio"),
    ("TASA_BADLAR", None, None, "BADLAR", "promedio"),
    ("TASA_TAMAR", None, None, "TAMAR", "promedio"),
    # el BCRA publica una sola serie de tasa de politica (empalmada); es una aproximacion para estos tres
    ("TASA_LEBAC35", None, None, "TASA_POLITICA", "promedio"),
    ("TASA_PASE7", None, None, "TASA_POLITICA", "promedio"),
    ("TASA_LELIQ", None, None, "TASA_POLITICA", "promedio"),
    ("PIB", "var. % i.a.", None, "PIB_IA_TRIM", "ultimo"),
    ("PIB", "var. % trim. s.e.", None, "PIB_TRIM_SE", "ultimo"),
    ("PIB", "var. % prom. anual", None, "PIB_PROM_ANUAL", "ultimo"),
    ("DESOCUPACION", None, None, "DESOCUPACION", "ultimo"),
    ("EXPORTACIONES", None, "mes", "EXPORTACIONES_MES", "ultimo"),
    ("EXPORTACIONES", None, "anio", "EXPORTACIONES_ANIO", "ultimo"),
    ("IMPORTACIONES", None, "mes", "IMPORTACIONES_MES", "ultimo"),
    ("IMPORTACIONES", None, "anio", "IMPORTACIONES_ANIO", "ultimo"),
]
NIVELES = ["TC_NOMINAL", "EXPORTACIONES", "IMPORTACIONES"]  # error_rel_pct solo para niveles
COLS = ["relevamiento", "variable", "unidad", "unidad_norm", "periodo_tipo", "fecha_objetivo",
        "horizonte_meses", "mediana", "promedio", "p10", "p90", "real", "error", "abs_error",
        "error_rel_pct", "dentro_p10_p90", "serie_real", "agregacion", "comparable"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out")
    a = ap.parse_args()
    data = Path(a.data_dir)
    out_path = Path(a.out) if a.out else data / "rem_errors.csv"

    rem = pd.read_csv(data / "rem_long.csv", parse_dates=["fecha_objetivo"])
    mon = pd.read_csv(data / "actuals_monthly.csv", parse_dates=["periodo"])
    if (data / "actuals_indec.csv").exists():
        mon = pd.concat([mon, pd.read_csv(data / "actuals_indec.csv", parse_dates=["periodo"])], ignore_index=True)
    else:
        print("AVISO: falta data/actuals_indec.csv (correr fetch_indec.py); se omiten las series INDEC.")
    cat = pd.read_csv(data / "actuals_catalog.csv", parse_dates=["hasta"])

    # meses incompletos: en series diarias se descarta el mes de la ultima observacion
    last_me = (cat.set_index("serie")["hasta"] + pd.offsets.MonthEnd(0)).dt.normalize()
    daily = set(cat.loc[cat["periodicidad"] == "D", "serie"])
    mon = mon[~(mon["serie"].isin(daily) & (mon["periodo"] >= mon["serie"].map(last_me)))]

    # duplicados en el REM (mismo objetivo cargado dos veces): no deberia pasar
    key = ["relevamiento", "variable", "unidad", "periodo_tipo", "fecha_objetivo"]
    dup = rem[rem.duplicated(key, keep=False)]
    if len(dup):
        print(f"AVISO: {len(dup)} filas duplicadas en rem_long.csv (bloques: {sorted(dup['bloque'].unique())}). "
              "Revisar map_variable() en build_data.py.")

    rem = rem[rem["variable"].isin({r[0] for r in RULES})].copy()
    rem["unidad_norm"] = rem["unidad"].str.replace("US$", "USD", regex=False)   # $/US$ y $/USD: mismo dato
    rem.loc[rem["variable"].str.startswith("TASA_"), "unidad_norm"] = "TNA; %"  # '%' (2016) y 'TNA; %'
    parts = []
    for var, pref, tipo, serie, col in RULES:
        m = rem["variable"].eq(var)
        if pref:
            m &= rem["unidad"].str.startswith(pref)
        if tipo:
            m &= rem["periodo_tipo"].eq(tipo)
        p = rem[m].copy()
        p["serie_real"], p["_col"] = serie, col
        parts.append(p)
    x = pd.concat(parts)

    res = []
    for col in ("promedio", "ultimo"):
        sub = x[x["_col"] == col]
        r = mon[["serie", "periodo", col]].rename(
            columns={"serie": "serie_real", "periodo": "fecha_objetivo", col: "real"})
        res.append(sub.merge(r, on=["serie_real", "fecha_objetivo"], how="left"))
    out = pd.concat(res)
    out["agregacion"] = out["_col"].map({"promedio": "promedio mensual", "ultimo": "dato del mes"})

    print("Filas REM sin real (todavia no publicado / sin serie):")
    print(out[out["real"].isna()].groupby("variable").size().to_string() or "  ninguna")
    out = out.dropna(subset=["real", "mediana"]).copy()

    out["error"] = out["real"] - out["mediana"]
    out["abs_error"] = out["error"].abs()
    out["error_rel_pct"] = np.where(out["variable"].isin(NIVELES), out["error"] / out["real"].abs() * 100, np.nan)
    ok = out["p10"].notna() & out["p90"].notna()
    out["dentro_p10_p90"] = ((out["real"] >= out["p10"]) & (out["real"] <= out["p90"])).where(ok).astype(float)

    # IPC nacional: INDEC no publicaba nivel nacional antes de 2017 (mensual) / dic-2017 (i.a.)
    nac = out["variable"] == "IPC_NG_NAC"
    mens = out["unidad"].str.contains("mensual")
    old = (mens & (out["fecha_objetivo"] < "2017-01-01")) | (~mens & (out["fecha_objetivo"] < "2017-12-01"))
    out["comparable"] = ~(nac & old)

    out = out.sort_values(["variable", "unidad_norm", "periodo_tipo", "fecha_objetivo", "relevamiento"])[COLS]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    g = out[out["comparable"]].groupby(["variable", "unidad_norm", "periodo_tipo"]).agg(
        n=("error", "size"), sesgo=("error", "mean"), MAE=("abs_error", "mean"),
        cobertura_p10_p90=("dentro_p10_p90", "mean"))
    print(f"\n{len(out)} filas en {out_path}  ({int((~out['comparable']).sum())} no comparables)\n")
    print(g.round(2).to_string())


if __name__ == "__main__":
    main()
