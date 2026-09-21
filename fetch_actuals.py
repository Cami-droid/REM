#!/usr/bin/env python
"""
fetch_actuals.py - Descarga series de resultados reales de la API BCRA Estadisticas v4.0.

Uso:
    pip install pandas requests
    python fetch_actuals.py --list                        # cuenta del catalogo
    python fetch_actuals.py --list badlar                 # filtra (todas las palabras deben aparecer)
    python fetch_actuals.py --list inflacion mensual --save data/bcra_catalogo.csv
    python fetch_actuals.py                               # baja las series de SERIES
    python fetch_actuals.py --series TC_MAYORISTA BADLAR --desde 2016-01-01
    python fetch_actuals.py --insecure                    # si falla la cadena SSL de api.bcra.gob.ar

IDs: los de SERIES vienen de la v3 y hay que confirmarlos con --list. Cada ID se valida contra el
catalogo (existe + palabras clave en la descripcion); si no coincide, la serie se saltea (usar --force
para bajarla igual). Para corregir un ID sin tocar el codigo: actuals_ids.json  {"BADLAR": 8}

Salidas (en data/):
    actuals_daily.csv     serie, fecha, valor           (tal como publica la API)
    actuals_monthly.csv   serie, periodo, promedio, ultimo, n_obs   (periodo = fin de mes)
    actuals_catalog.csv   serie, id, descripcion, unidad, periodicidad, rango, descargado

Nota: 'promedio' sirve para TC/tasas (REM pide promedio mensual); 'ultimo' para IPC (1 dato por mes).
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
API = "https://api.bcra.gob.ar/estadisticas/v4.0/Monetarias"
PAGE = 3000  # maximo de la API

# nombre: (id por defecto, palabras que deben aparecer en la descripcion)
SERIES = {
    "IPC_MENSUAL": (27, ["inflacion", "mensual"]),
    "IPC_INTERANUAL": (28, ["inflacion", "interanual"]),
    "TC_MAYORISTA": (5, ["mayorista"]),
    "BADLAR": (7, ["badlar"]),
    "TAMAR": (44, ["tamar"]),
    "TASA_POLITICA": (160, ["politica monetaria"]),  # TNA; el 161 es la misma serie en TEA. Termina 2025-07-10
}


def norm(x) -> str:
    s = unicodedata.normalize("NFD", str(x or "").strip().lower())
    return re.sub(r"\s+", " ", "".join(c for c in s if unicodedata.category(c) != "Mn"))


# ----------------------------------------------------------------------------
# HTTP
# ----------------------------------------------------------------------------
class Api:
    def __init__(self, insecure=False):
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": "rem-app/1.0", "Accept": "application/json"})
        self.verify = not insecure
        if insecure:
            import urllib3
            urllib3.disable_warnings()

    def get(self, url, params=None):
        last = None
        for i in range(4):
            try:
                r = self.s.get(url, params=params, timeout=40, verify=self.verify)
            except requests.exceptions.SSLError as e:
                sys.exit(f"Error SSL con api.bcra.gob.ar. Reintentar con --insecure.\n{e}")
            except requests.RequestException as e:
                last = e
                time.sleep(2 ** i)
                continue
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 500, 502, 503, 504):
                last = f"HTTP {r.status_code}"
                time.sleep(2 ** i)
                continue
            raise RuntimeError(f"HTTP {r.status_code} en {r.url}: {r.text[:200]}")
        raise RuntimeError(f"Fallo tras reintentos: {last}")

    def paged(self, url, params, extract):
        items, off = [], 0
        while True:
            js = self.get(url, {**params, "limit": PAGE, "offset": off})
            chunk = extract(js.get("results") or [])
            items.extend(chunk)
            if len(chunk) < PAGE:
                return items
            off += PAGE

    def catalog(self) -> pd.DataFrame:
        rows = self.paged(API, {}, lambda res: list(res))
        return pd.DataFrame(rows)

    def series(self, id_, desde, hasta) -> pd.DataFrame:
        rows = self.paged(f"{API}/{id_}", {"desde": desde, "hasta": hasta},
                          lambda res: [d for r in res for d in (r.get("detalle") or [])])
        df = pd.DataFrame(rows, columns=["fecha", "valor"])
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
        df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
        return df.dropna().drop_duplicates("fecha").sort_values("fecha").reset_index(drop=True)


# ----------------------------------------------------------------------------
# Comandos
# ----------------------------------------------------------------------------
def cmd_list(api, words, save, max_rows=80):
    cat = api.catalog()
    print(f"Catalogo: {len(cat)} series")
    if words:
        cols_txt = [c for c in ("descripcion", "categoria", "tipoSerie") if c in cat.columns]
        blob = cat[cols_txt].fillna("").astype(str).agg(" ".join, axis=1).map(norm)
        for w in map(norm, words):
            cat = cat[blob.loc[cat.index].str.contains(re.escape(w))]
    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        cat.to_csv(save, index=False, encoding="utf-8")
        print(f"Guardado: {save} ({len(cat)} filas)")
    if not words and not save:
        print("Usar --list <palabras> para filtrar, p.ej.: --list inflacion mensual")
        return
    cols = [c for c in ("idVariable", "periodicidad", "unidadExpresion", "primerFechaInformada",
                        "ultFechaInformada", "descripcion") if c in cat.columns]
    print(f"{len(cat)} coincidencias" + (f" (se muestran {max_rows})" if len(cat) > max_rows else ""))
    print(cat[cols].head(max_rows).to_string(index=False, max_colwidth=90))


def to_monthly(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["periodo"] = d["fecha"].dt.to_period("M").dt.end_time.dt.normalize()
    g = d.groupby("periodo")["valor"]
    return pd.DataFrame({"promedio": g.mean(), "ultimo": g.last(), "n_obs": g.count()}).reset_index()


def cmd_fetch(api, names, desde, force):
    ids_file = ROOT / "actuals_ids.json"
    over = json.loads(ids_file.read_text(encoding="utf-8")) if ids_file.exists() else {}
    cat = api.catalog().set_index("idVariable")
    hasta = dt.date.today().isoformat()
    daily, monthly, meta, fallidas = [], [], [], []

    for name in names:
        id_, kws = SERIES[name]
        id_ = int(over.get(name, id_))
        if id_ not in cat.index:
            print(f"SALTEADA {name}: id {id_} no existe en el catalogo. Buscar con --list")
            fallidas.append(name)
            continue
        row = cat.loc[id_]
        desc = row.get("descripcion", "")
        if not all(norm(k) in norm(desc) for k in kws) and not force:
            print(f"SALTEADA {name}: id {id_} = '{desc}' no contiene {kws}. "
                  f"Corregir con --list / actuals_ids.json (o --force)")
            fallidas.append(name)
            continue
        df = api.series(id_, desde, hasta)
        if df.empty:
            print(f"VACIA    {name} (id {id_}) sin datos en {desde}..{hasta}")
            fallidas.append(name)
            continue
        df.insert(0, "serie", name)
        daily.append(df)
        m = to_monthly(df)
        m.insert(0, "serie", name)
        monthly.append(m)
        meta.append({"serie": name, "id": id_, "descripcion": desc,
                     "unidad": row.get("unidadExpresion"), "periodicidad": row.get("periodicidad"),
                     "desde": df["fecha"].min().date(), "hasta": df["fecha"].max().date(),
                     "n": len(df), "descargado": hasta})
        print(f"OK       {name:15s} id {id_:<4d} {len(df):5d} obs  "
              f"{df['fecha'].min():%Y-%m-%d}..{df['fecha'].max():%Y-%m-%d}  {desc[:60]}")

    if not daily:
        sys.exit("No se bajo ninguna serie.")
    OUT.mkdir(parents=True, exist_ok=True)
    pd.concat(daily).to_csv(OUT / "actuals_daily.csv", index=False)
    pd.concat(monthly).to_csv(OUT / "actuals_monthly.csv", index=False)
    pd.DataFrame(meta).to_csv(OUT / "actuals_catalog.csv", index=False, encoding="utf-8")
    print(f"\nListo: data/actuals_daily.csv, actuals_monthly.csv, actuals_catalog.csv")
    if fallidas:
        print(f"Series sin bajar: {fallidas}")
        sys.exit(1)  # falla el Action si hay series rotas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", nargs="*", metavar="TEXTO", help="lista/filtra el catalogo y sale")
    ap.add_argument("--save", help="con --list: guarda el resultado en CSV")
    ap.add_argument("--series", nargs="*", choices=list(SERIES), default=list(SERIES))
    ap.add_argument("--desde", default="2016-01-01")
    ap.add_argument("--force", action="store_true", help="bajar aunque la descripcion no coincida")
    ap.add_argument("--insecure", action="store_true", help="no verificar certificado SSL")
    a = ap.parse_args()

    api = Api(insecure=a.insecure)
    if a.list is not None:
        cmd_list(api, a.list, a.save)
    else:
        cmd_fetch(api, a.series, a.desde, a.force)


if __name__ == "__main__":
    main()
