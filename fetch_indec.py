#!/usr/bin/env python
"""
fetch_indec.py - Resultados reales de INDEC (via API de Series de Tiempo, datos.gob.ar) en el mismo
formato que actuals_monthly.csv, para cruzar con el REM.

Uso:
    pip install pandas requests
    python fetch_indec.py --list nucleo gba                # busca series por texto (para hallar IDs)
    python fetch_indec.py                                  # baja y deriva todo
    python fetch_indec.py --only PIB_ORIG PIB_SA           # solo esas series base
    python fetch_indec.py --start 2014-01-01

Cada serie base tiene un id de partida y palabras clave. Si el id no existe o la descripcion no cumple,
busca por texto: si hay UN solo candidato lo usa; si hay varios, los lista y la serie queda sin bajar.
Para fijar un id sin tocar el codigo: indec_ids.json   {"PIB_SA": "3.2_..."}  (un id fijado se usa sin validar)

Salidas (data/):
    actuals_indec.csv   serie, periodo (fin de periodo), promedio, ultimo, n_obs   (mismo esquema que BCRA)
    indec_catalog.csv   serie base, id, descripcion, unidad, frecuencia, desde, hasta, n, descargado

Series derivadas (nombre -> lo que compara compute_errors.py):
    IPC_NUCLEO_MENSUAL/INTERANUAL, IPC_GBA_MENSUAL/INTERANUAL, IPC_GBA_NUCLEO_MENSUAL/INTERANUAL  (de indices)
    PIB_IA_TRIM (i.a. trimestral, original) | PIB_TRIM_SE (q/q desestacionalizado) | PIB_PROM_ANUAL (prom. anual)
    DESOCUPACION (% PEA, trimestral) | EXPORTACIONES_MES/ANIO, IMPORTACIONES_MES/ANIO (millones de USD)
El PIB se revisa: se usa siempre la ultima version publicada, no la primera estimacion.
"""
import argparse
import datetime as dt
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).parent
OUT = ROOT / "data"
API = "https://apis.datos.gob.ar/series/api"
FREQ = {"R/P1M": "M", "R/P3M": "Q", "R/P1Y": "A"}
PER = {"M": "M", "Q": "Q", "A": "Y"}

# nombre: id de partida (a confirmar con --list), texto de busqueda, palabras clave (todas), excluir, frecuencia
SPECS = {
    "NUCLEO_NAC_IDX": dict(id="148.3_INUCLEONAL_DICI_M_19", q="IPC nucleo nacional base dic 2016 mensual",
                           kw=["nucleo", "nacional"], freq="M"),
    "NG_GBA_IDX": dict(id="103.1_I2N_2016_M_19", q="IPC-GBA nivel general base abr 2016 mensual",
                       kw=["gba", "nivel general"], freq="M"),
    "NUCLEO_GBA_IDX": dict(id="103.1_I2N_2016_M_15", q="IPC-GBA nucleo base abr 2016 mensual",
                           kw=["gba", "nucleo"], freq="M"),
    "PIB_ORIG": dict(id="4.2_OGP_2004_T_17", q="PIB precios de comprador millones de pesos de 2004 trimestral",
                     kw=["pib", "2004", "trimestral"], kw_not=["desestacionaliz", "implicit"], freq="Q"),
    # id por analogia con 1.1_OGP_D_1993_A_17 (OGP = PIB); si falla: --list oferta demanda desestacionalizado 2004
    "PIB_SA": dict(id="3.2_OGP_D_2004_T_17", q=["PIB desestacionalizado millones de pesos de 2004 trimestral",
                               "oferta demanda globales desestacionalizados base 2004 trimestral"],
                   kw=["desestacionaliz", ("pib", "producto interno bruto"), "2004", "trimestral"],
                   kw_not=["implicit"], freq="Q"),
    "DESOCUPACION": dict(id="45.2_ECTDT_0_T_33", q="tasa de desempleo total EPH trimestral",
                         kw=[("tasa de desocupacion", "tasa de desempleo"), "total"], freq="Q"),
    "EXPORTACIONES": dict(id="74.3_IET_0_M_16", q="exportaciones intercambio comercial argentino mensual",
                          kw=["exportaciones"], freq="M"),
    "IMPORTACIONES": dict(id="74.3_IIT_0_M_25", q="importaciones intercambio comercial argentino mensual",
                          kw=["importaciones"], freq="M"),
}


def norm(x) -> str:
    s = unicodedata.normalize("NFD", str(x or "").strip().lower())
    return re.sub(r"\s+", " ", "".join(c for c in s if unicodedata.category(c) != "Mn"))


def matches(spec, text) -> bool:
    """kw: todas deben aparecer; un elemento tupla vale como 'alguna de'. kw_not: ninguna debe aparecer."""
    t = norm(text)
    ok = all(any(norm(a) in t for a in (k if isinstance(k, (tuple, list)) else [k])) for k in spec["kw"])
    return ok and not any(norm(k) in t for k in spec.get("kw_not", []))


# ----------------------------------------------------------------------------
# API
# ----------------------------------------------------------------------------
class Api:
    def __init__(self):
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": "rem-app/1.0"})

    def get(self, path, params, soft=False):
        last = None
        for i in range(4):
            try:
                r = self.s.get(f"{API}/{path}", params=params, timeout=40)
            except requests.RequestException as e:
                last = e
                time.sleep(2 ** i)
                continue
            if r.status_code == 200:
                return r.json()
            if soft and r.status_code in (400, 404):
                return None
            if r.status_code in (429, 500, 502, 503, 504):
                last = f"HTTP {r.status_code}"
                time.sleep(2 ** i)
                continue
            raise RuntimeError(f"HTTP {r.status_code} en {r.url}: {r.text[:200]}")
        raise RuntimeError(f"Fallo tras reintentos: {last}")

    def search(self, q, limit=100):
        js = self.get("search/", {"q": q, "limit": limit}) or {}
        return js.get("data", [])

    def series(self, id_, start, default_freq):
        """-> dict(id, desc, units, freq, s) con s indexada por fin de periodo, o None si el id no existe."""
        js = self.get("series/", {"ids": id_, "start_date": start, "limit": 1000, "sort": "asc",
                                  "format": "json", "metadata": "full"}, soft=True)
        if not js or not js.get("data"):
            return None
        meta = next((m for m in js.get("meta", []) if isinstance(m, dict) and "field" in m), {})
        field, dset = meta.get("field", {}), meta.get("dataset", {})
        f = FREQ.get(field.get("frequency"), default_freq)
        df = pd.DataFrame(js["data"], columns=["fecha", "valor"])
        df["fecha"] = pd.to_datetime(df["fecha"])
        df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
        df = df.dropna()
        add = {"M": 0, "Q": 2, "A": 11}[f]
        idx = (df["fecha"] + pd.DateOffset(months=add)) + pd.offsets.MonthEnd(0)
        s = pd.Series(df["valor"].values, index=idx.dt.normalize().values).sort_index()
        return {"id": id_, "desc": f"{field.get('description', '')} | {dset.get('title', '')}",
                "units": field.get("units", ""), "freq": f, "s": s}


# ----------------------------------------------------------------------------
# Resolucion de series base
# ----------------------------------------------------------------------------
def resolve(api, name, spec, over, start):
    id_ = over.get(name, spec.get("id"))
    if id_:
        got = api.series(id_, start, spec["freq"])
        if got and (name in over or matches(spec, got["desc"])):  # id fijado en indec_ids.json: se confia
            return got
        why = "no existe" if not got else f"descripcion '{got['desc'][:70]}' no cumple {spec['kw']}"
        print(f"  {name}: id {id_} {why}; busco por texto...")
    hits, seen = [], set()
    qs = spec["q"] if isinstance(spec["q"], list) else [spec["q"]]
    for h in (h for q in qs for h in api.search(q)):
        fld, dset = h.get("field", {}), h.get("dataset", {})
        text = f"{fld.get('description', '')} {dset.get('title', '')}"
        f = FREQ.get(fld.get("frequency"))
        if fld.get("id") not in seen and matches(spec, text) and (f is None or f == spec["freq"]):
            seen.add(fld.get("id"))
            hits.append((fld.get("id"), fld.get("description", ""), fld.get("time_index_start"),
                         fld.get("time_index_end")))
    if len(hits) == 1:
        got = api.series(hits[0][0], start, spec["freq"])
        if got:
            print(f"  {name}: resuelta por texto -> {hits[0][0]}  (confirmar y fijar en indec_ids.json)")
            return got
    print(f"SIN RESOLVER {name}: {len(hits)} candidatos. Buscar con --list <palabras> y fijar el id en indec_ids.json")
    for h in hits[:8]:
        print(f"      {h[0]}  {h[2]}..{h[3]}  {h[1][:80]}")
    return None


# ----------------------------------------------------------------------------
# Derivadas (todo en PeriodIndex para que los rezagos sean por periodo, no por fila)
# ----------------------------------------------------------------------------
def per(s, f):
    p = s.copy()
    p.index = pd.DatetimeIndex(p.index).to_period(PER[f])
    return p.reindex(pd.period_range(p.index.min(), p.index.max(), freq=PER[f]))


def pct(p, lag):
    return (p / p.shift(lag) - 1) * 100


def frame(name, p):
    p = p.dropna()
    if p.empty:
        return None
    if isinstance(p.index, pd.PeriodIndex):
        fechas = p.index.end_time.normalize()
    else:  # anios
        fechas = pd.to_datetime([f"{y}-12-31" for y in p.index])
    return pd.DataFrame({"serie": name, "periodo": fechas, "promedio": p.values, "ultimo": p.values, "n_obs": 1})


def by_year(p, how, n):
    g = p.groupby(p.index.year).agg([how, "count"])
    a = g[how].where(g["count"] == n)
    return a.reindex(range(a.index.min(), a.index.max() + 1))


def derive(base):
    out = []

    def add(name, p):
        f = frame(name, p)
        if f is not None:
            out.append(f)

    for key, tag in (("NUCLEO_NAC_IDX", "IPC_NUCLEO"), ("NG_GBA_IDX", "IPC_GBA"), ("NUCLEO_GBA_IDX", "IPC_GBA_NUCLEO")):
        if key in base:
            p = per(base[key]["s"], "M")
            add(f"{tag}_MENSUAL", pct(p, 1))
            add(f"{tag}_INTERANUAL", pct(p, 12))
    if "PIB_ORIG" in base:
        q = per(base["PIB_ORIG"]["s"], "Q")
        add("PIB_IA_TRIM", pct(q, 4))
        a = by_year(q, "mean", 4)
        add("PIB_PROM_ANUAL", (a / a.shift(1) - 1) * 100)
    if "PIB_SA" in base:
        add("PIB_TRIM_SE", pct(per(base["PIB_SA"]["s"], "Q"), 1))
    if "DESOCUPACION" in base:
        d = per(base["DESOCUPACION"]["s"], "Q")
        if d.max() < 1.5:  # viene como fraccion
            print("  DESOCUPACION: valores < 1.5, se multiplican por 100 (fraccion -> %)")
            d = d * 100
        add("DESOCUPACION", d)
    for key, tag in (("EXPORTACIONES", "EXPORTACIONES"), ("IMPORTACIONES", "IMPORTACIONES")):
        if key in base:
            m = per(base[key]["s"], "M")
            add(f"{tag}_MES", m)
            add(f"{tag}_ANIO", by_year(m, "sum", 12))
    return out


# ----------------------------------------------------------------------------
def cmd_list(api, words):
    hits = api.search(" ".join(words), limit=60)
    print(f"{len(hits)} resultados")
    for h in hits:
        f, d = h.get("field", {}), h.get("dataset", {})
        print(f"{f.get('id', ''):34s} {str(f.get('time_index_start', ''))[:7]}..{str(f.get('time_index_end', ''))[:7]}  "
              f"{FREQ.get(f.get('frequency'), '?')}  {str(f.get('units', ''))[:22]:22s} "
              f"{f.get('description', '')[:70]} | {d.get('title', '')[:50]}")


def cmd_fetch(api, only, start):
    ids_file = ROOT / "indec_ids.json"
    over = json.loads(ids_file.read_text(encoding="utf-8")) if ids_file.exists() else {}
    base, meta, fallidas = {}, [], []
    for name in only:
        got = resolve(api, name, SPECS[name], over, start)
        if not got:
            fallidas.append(name)
            continue
        base[name] = got
        s = got["s"]
        meta.append({"serie": name, "id": got["id"], "descripcion": got["desc"], "unidad": got["units"],
                     "frecuencia": got["freq"], "desde": s.index.min().date(), "hasta": s.index.max().date(),
                     "n": len(s), "descargado": dt.date.today().isoformat()})
        print(f"OK       {name:15s} {got['id']:30s} {len(s):4d} obs {s.index.min():%Y-%m}..{s.index.max():%Y-%m}  "
              f"{got['desc'][:50]}")
    frames = derive(base)
    if not frames:
        sys.exit("No se pudo derivar ninguna serie.")
    res = pd.concat(frames)
    OUT.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT / "actuals_indec.csv", index=False)
    pd.DataFrame(meta).to_csv(OUT / "indec_catalog.csv", index=False, encoding="utf-8")
    print("\nDerivadas:")
    for n, g in res.groupby("serie"):
        print(f"  {n:26s} {len(g):4d} obs  {g['periodo'].min():%Y-%m}..{g['periodo'].max():%Y-%m}")
    print("\nListo: data/actuals_indec.csv, data/indec_catalog.csv")
    if fallidas:
        print(f"Series base sin bajar: {fallidas}")
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", nargs="+", metavar="TEXTO", help="busca series por texto y sale")
    ap.add_argument("--only", nargs="*", choices=list(SPECS), default=list(SPECS))
    ap.add_argument("--start", default="2014-01-01")
    a = ap.parse_args()
    api = Api()
    if a.list:
        cmd_list(api, a.list)
    else:
        cmd_fetch(api, a.only, a.start)


if __name__ == "__main__":
    main()
