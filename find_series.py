#!/usr/bin/env python
"""
find_series.py - Ayuda local para fijar ids de series de datos.gob.ar que fetch_indec.py no pudo resolver.

Uso:
    python find_series.py pib_sa                      # busca el PIB desestacionalizado y lo valida contra cifras oficiales
    python find_series.py pib_sa --pin 3.2_OGP_D_2004_T_17   # valida un id concreto y lo fija en indec_ids.json
    python find_series.py pib_sa --no-write           # solo muestra el ranking
    python find_series.py spnf                        # lista candidatas de resultado primario del SPNF (Hacienda)
    python find_series.py buscar oferta demanda       # busqueda libre (id, frecuencia, rango, unidades, descripcion)

pib_sa: prueba ids 3.2_OGP_D_2004_T_1..40 mas lo que devuelva la busqueda, calcula la variacion trimestral (q/q) de
cada serie trimestral y la compara con las variaciones oficiales del INDEC (OFFICIAL_QQ). Si UNA serie coincide
claramente, la escribe en indec_ids.json ({"PIB_SA": "<id>"}) junto a fetch_indec.py; despues correr fetch_indec.py.
Ademas valida el NIVEL contra la serie original del PIB (4.2_OGP_2004_T_17): una serie desestacionalizada del PIB tiene
casi el mismo nivel anual (ratio ~1,00); "oferta global" (PIB + importaciones) da ~1,2-1,3 y se descarta. Tambien se
descartan series que no sean realmente trimestrales (p. ej. el EMAE mensual).
OFFICIAL_QQ son primeras publicaciones (el INDEC revisa la serie): la tolerancia es de +-0,5 pp en promedio.
Podes agregar mas trimestres de los "Informes de avance del nivel de actividad".
"""
import argparse
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).parent
API = "https://apis.datos.gob.ar/series/api"
FREQ = {"R/P1M": "M", "R/P3M": "Q", "R/P1Y": "A"}
# variacion % trimestral desestacionalizada del PIB, segun informes del INDEC (trimestre: q/q)
OFFICIAL_QQ = {"2024Q3": 3.9, "2025Q4": 0.6, "2026Q1": 0.7}
ORIG_ID = "4.2_OGP_2004_T_17"   # PIB original trimestral (el que ya baja fetch_indec.py)
RATIO_TOL = 0.03   # el nivel SA / original debe estar en 1 +- 3%
TOL = 0.5          # pp: diferencia media maxima para aceptar
GAP = 0.2          # pp: ventaja minima del mejor sobre el segundo (si no, es ambiguo)
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "rem-app/1.0"})


def norm(x) -> str:
    s = unicodedata.normalize("NFD", str(x or "").strip().lower())
    return re.sub(r"\s+", " ", "".join(c for c in s if unicodedata.category(c) != "Mn"))


def get(path, params, soft=False):
    last = None
    for i in range(4):
        try:
            r = SESSION.get(f"{API}/{path}", params=params, timeout=40)
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


def search(q, limit=100):
    return (get("search/", {"q": q, "limit": limit}) or {}).get("data", [])


def fetch_series(id_, start="2023-01-01"):
    """-> dict(id, desc, dataset, units, freq, s) o None si el id no existe / no tiene datos."""
    js = get("series/", {"ids": id_, "start_date": start, "limit": 1000, "sort": "asc",
                         "format": "json", "metadata": "full"}, soft=True)
    if not js or not js.get("data"):
        return None
    meta = next((m for m in js.get("meta", []) if isinstance(m, dict) and "field" in m), {})
    field, dset = meta.get("field", {}), meta.get("dataset", {})
    df = pd.DataFrame(js["data"], columns=["fecha", "valor"])
    df["fecha"] = pd.to_datetime(df["fecha"])
    df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
    df = df.dropna()
    if df.empty:
        return None
    return {"id": id_, "desc": field.get("description", ""), "dataset": dset.get("title", ""),
            "units": field.get("units", ""), "freq": FREQ.get(field.get("frequency"), "?"),
            "s": pd.Series(df["valor"].values, index=df["fecha"].values)}


def really_quarterly(s):
    d = pd.Series(pd.DatetimeIndex(s.index)).diff().dt.days.dropna()
    return len(d) > 0 and d.median() >= 80


def level_ratio(s, orig):
    """mediana de (serie candidata / PIB original) en los trimestres en comun"""
    a, b = s.copy(), orig.copy()
    a.index = pd.DatetimeIndex(a.index).to_period("Q")
    b.index = pd.DatetimeIndex(b.index).to_period("Q")
    r = (a / b).dropna()
    return float(r.median()) if len(r) else float("nan")


def qq(s):
    p = s.copy()
    p.index = pd.DatetimeIndex(p.index).to_period("Q")
    p = p[~p.index.duplicated()].sort_index()
    p = p.reindex(pd.period_range(p.index.min(), p.index.max(), freq="Q"))
    return (p / p.shift(1) - 1) * 100


def score(got):
    """(diferencia media absoluta vs. cifras oficiales, trimestres comparados, ultimo q/q)"""
    g = qq(got["s"])
    diffs = [abs(g[pd.Period(k)] - v) for k, v in OFFICIAL_QQ.items()
             if pd.Period(k) in g.index and pd.notna(g[pd.Period(k)])]
    last = g.dropna().iloc[-1] if g.notna().any() else float("nan")
    return (sum(diffs) / len(diffs) if diffs else float("inf")), len(diffs), last


def write_id(name, id_):
    f = ROOT / "indec_ids.json"
    d = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
    d[name] = id_
    f.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nEscrito {f.name}: {name} = {id_}. Ahora correr: python fetch_indec.py")


# ----------------------------------------------------------------------------
def cmd_pib_sa(pin, write):
    if pin:
        got = fetch_series(pin)
        if not got:
            sys.exit(f"El id {pin} no existe o no tiene datos.")
        cands = [got]
    else:
        ids = [f"3.2_OGP_D_2004_T_{n}" for n in range(1, 41)]
        for q in ("PIB desestacionalizado trimestral base 2004", "oferta demanda globales desestacionalizado 2004 trimestral",
                  "producto interno bruto desestacionalizado", "PIB precios de comprador desestacionalizado"):
            for h in search(q):
                fid = h.get("field", {}).get("id")
                if fid and fid not in ids and FREQ.get(h.get("field", {}).get("frequency")) == "Q" \
                        and ("desestacionaliz" in norm(h.get("field", {}).get("description", "") + h.get("dataset", {}).get("title", ""))):
                    ids.append(fid)
        print(f"Probando {len(ids)} ids candidatos...")
        cands = [c for c in (fetch_series(i) for i in ids)
                 if c and c["freq"] == "Q" and really_quarterly(c["s"])]
    orig = fetch_series(ORIG_ID)
    if orig is None:
        print(f"AVISO: no se pudo bajar {ORIG_ID}; se omite la validacion de nivel.")
    rows = []
    for c in cands:
        d, n, last = score(c)
        ratio = level_ratio(c["s"], orig["s"]) if orig else float("nan")
        nivel_ok = (abs(ratio - 1) <= RATIO_TOL) if ratio == ratio else True
        rows.append((d, n, last, c, ratio, nivel_ok))
    rows.sort(key=lambda r: (not r[5], r[0], -r[1]))
    print(f"\n{'id':30s} {'dif.media':>9s} {'n':>2s} {'ult.q/q':>8s} {'nivel/orig':>10s}  descripcion | dataset")
    for d, n, last, c, ratio, ok in rows[:15]:
        dm = f"{d:9.2f}" if d != float("inf") else "      s/d"
        print(f"{c['id']:30s} {dm} {n:2d} {last:8.2f} {ratio:10.3f}{'' if ok else ' X'}  "
              f"{c['desc'][:55]} | {c['dataset'][:35]}")
    print("(nivel/orig ~1,00 = mismo nivel que el PIB; 'X' = nivel incompatible con el PIB, se descarta)")
    rows = [r for r in rows if r[5]] or rows
    rows = [(d, n, last, c) for d, n, last, c, _, _ in rows]
    print(f"\nOficial (q/q %): {OFFICIAL_QQ}")
    if not rows or rows[0][0] == float("inf"):
        sys.exit("Ninguna serie trimestral con datos comparables. Probar: python find_series.py buscar pib desestacionalizado")
    best = rows[0]
    ambiguo = len(rows) > 1 and rows[1][0] - best[0] < GAP and not pin
    if best[0] <= TOL and not ambiguo:
        print(f"\nCOINCIDE: {best[3]['id']} (dif. media {best[0]:.2f} pp)  {best[3]['desc'][:80]}")
        if write:
            write_id("PIB_SA", best[3]["id"])
    else:
        print("\nNo hay una coincidencia clara" + (" (dos series muy parecidas)" if ambiguo else "") +
              f" (umbral {TOL} pp). Revisar la tabla; para fijar una a mano: python find_series.py pib_sa --pin <id>")
        if pin and write:
            print(f"AVISO: se fija {pin} por pedido explicito aunque la diferencia es {best[0]:.2f} pp.")
            write_id("PIB_SA", pin)


def cmd_list(words, limit=40):
    seen = set()
    hits = search(" ".join(words), limit=limit)
    print(f"{len(hits)} resultados")
    for h in hits:
        f, d = h.get("field", {}), h.get("dataset", {})
        if f.get("id") in seen:
            continue
        seen.add(f.get("id"))
        print(f"{f.get('id', ''):34s} {FREQ.get(f.get('frequency'), '?')} "
              f"{str(f.get('time_index_start', ''))[:7]}..{str(f.get('time_index_end', ''))[:7]}  "
              f"{str(f.get('units', ''))[:24]:24s} {f.get('description', '')[:70]} | {d.get('title', '')[:45]}")


def cmd_spnf():
    print("Candidatas para resultado primario del SPNF (el REM lo publica ANUAL, en miles de millones de $; "
          "el real debe ser el acumulado del anio, base caja):\n")
    for q in ("resultado primario sector publico nacional no financiero",
              "esquema ahorro inversion financiamiento resultado primario",
              "resultado primario SPNF"):
        cmd_list(q.split(), limit=25)
        print()


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pib_sa")
    p.add_argument("--pin")
    p.add_argument("--no-write", action="store_true")
    sub.add_parser("spnf")
    b = sub.add_parser("buscar")
    b.add_argument("words", nargs="+")
    a = ap.parse_args()
    if a.cmd == "pib_sa":
        cmd_pib_sa(a.pin, not a.no_write)
    elif a.cmd == "spnf":
        cmd_spnf()
    else:
        cmd_list(a.words)


if __name__ == "__main__":
    main()
