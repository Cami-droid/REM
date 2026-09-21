#!/usr/bin/env python
"""
build_data.py - Descarga y parsea los xlsx del REM (BCRA) a una tabla normalizada.

Uso:
    pip install pandas openpyxl requests
    python build_data.py                          # download + parse
    python build_data.py download [--start-year 2016]
    python build_data.py parse [--raw-dir data/raw] [--out-dir data]
    python build_data.py inspect data/raw/rem_2026-08.xlsx

Nombres de archivo que entiende (en data/raw/):
    rem_YYYY-MM.xlsx   |   REMyymmdd_Tablas_web.xlsx   |   tablas-relevamiento-expectativas-mercado-ago-2026.xlsx
(el mes de relevamiento se toma primero del titulo de la hoja; el nombre es respaldo)

Salidas (en data/):
    rem_long.csv / .parquet   una fila por (relevamiento, variable, periodo objetivo) con todos los estadisticos
    structure_report.txt      resumen por archivo: bloques detectados, variables sin mapear, avisos
    missing.txt               meses que no se pudieron bajar
"""
import argparse
import datetime as dt
import re
import time
import unicodedata
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data"

BASE = "https://www.bcra.gob.ar/archivos/Pdfs/PublicacionesEstadisticas/informes/"   # estilo nuevo
BASE_OLD = "https://www.bcra.gob.ar/archivos/Pdfs/PublicacionesEstadisticas/"        # estilo viejo
MESES = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]
MESES_FULL = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio",
              "agosto", "septiembre", "octubre", "noviembre", "diciembre"]
MES_IDX = {m: i + 1 for i, m in enumerate(MESES)}
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"}


# ----------------------------------------------------------------------------
# Utilidades de fechas / texto
# ----------------------------------------------------------------------------
def norm(x) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    s = str(x).strip().lower()
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", s)


def month_end(y: int, m: int) -> pd.Timestamp:
    return (pd.Timestamp(year=y, month=m, day=1) + pd.offsets.MonthEnd(0)).normalize()


def parse_mmm_yy(s):
    m = re.fullmatch(r"([a-z]{3,4})[-\s/.]?(\d{2}|\d{4})", norm(s))
    if not m:
        return None
    mes = MES_IDX.get(m.group(1)[:3])
    if not mes:
        return None
    y = int(m.group(2))
    y += 2000 if y < 100 else 0
    return month_end(y, mes)


def parse_trim(s):
    m = re.fullmatch(r"trim\.?\s*(iv|i{1,3})[-\s]*(\d{2}|\d{4})", norm(s))
    if not m:
        return None
    q = {"i": 1, "ii": 2, "iii": 3, "iv": 4}[m.group(1)]
    y = int(m.group(2))
    y += 2000 if y < 100 else 0
    return month_end(y, q * 3)


def parse_date_txt(s):
    return parse_trim(s) or parse_mmm_yy(s)


# ----------------------------------------------------------------------------
# Descarga
# ----------------------------------------------------------------------------
def is_xlsx(b: bytes) -> bool:
    return b[:4] == b"PK\x03\x04"


def try_get(session, url):
    try:
        r = session.get(url, timeout=40)
    except requests.RequestException:
        return None
    if r.status_code == 200 and is_xlsx(r.content):
        return r.content
    return None


def last_weekdays(y, m, k=6):
    d = month_end(y, m).date()
    out = []
    while len(out) < k:
        if d.weekday() < 5:
            out.append(d)
        d -= dt.timedelta(days=1)
    return out


def candidate_urls(y, m, style):
    if style == "old":  # .../PublicacionesEstadisticas/REM220331%20Tablas%20web.xlsx (ultimo dia habil)
        out = []
        for d in last_weekdays(y, m):
            out.append(f"{BASE_OLD}REM{d:%y%m%d}%20Tablas%20web.xlsx")
            out.append(f"{BASE_OLD}REM{d:%y%m%d}_Tablas_web.xlsx")
        return out
    # estilo "tablas-...": desde sep-2023 en PublicacionesEstadisticas/ con año de 2 digitos (sep-23);
    # el de ago-2026 esta en informes/ con 4 digitos. Se prueban ambas combinaciones.
    combos = [(BASE_OLD, f"{y % 100:02d}", MESES[m - 1]), (BASE, str(y), MESES[m - 1]),
              (BASE_OLD, str(y), MESES[m - 1]), (BASE, f"{y % 100:02d}", MESES[m - 1]),
              (BASE_OLD, f"{y % 100:02d}", MESES_FULL[m - 1]), (BASE, str(y), MESES_FULL[m - 1])]
    return [f"{b}tablas-relevamiento-expectativas-mercado-{mes}-{yy}.xlsx" for b, yy, mes in combos]


def download(start_year: int, exclude_years):
    RAW.mkdir(parents=True, exist_ok=True)
    today = dt.date.today()
    s = requests.Session()
    s.headers.update(UA)

    extra = {}
    ef = ROOT / "extra_urls.txt"  # lineas "YYYY-MM URL"
    if ef.exists():
        for line in ef.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) == 2 and re.fullmatch(r"\d{4}-\d{2}", parts[0]):
                extra[parts[0]] = parts[1]

    missing, style = [], "old"
    for y in range(start_year, today.year + 1):
        if y in exclude_years:
            continue
        for m in range(1, 13):
            if (y, m) > (today.year, today.month):
                break
            key = f"{y}-{m:02d}"
            dest = RAW / f"rem_{key}.xlsx"
            if dest.exists() and dest.stat().st_size > 0:
                continue
            order = [style, "new" if style == "old" else "old"]
            content = None
            urls = [(None, extra[key])] if key in extra else []
            urls += [(st, u) for st in order for u in candidate_urls(y, m, st)]
            for st, u in urls:
                content = try_get(s, u)
                time.sleep(0.1)
                if content:
                    style = st or style
                    break
            if content:
                dest.write_bytes(content)
                print(f"OK      {key}")
            else:
                print(f"FALTA   {key}")
                missing.append(key)
    OUT.mkdir(exist_ok=True)
    (OUT / "missing.txt").write_text("\n".join(missing), encoding="utf-8")
    print(f"\nDescarga terminada. Faltantes: {len(missing)} (ver data/missing.txt)")


# ----------------------------------------------------------------------------
# Parseo
# ----------------------------------------------------------------------------
def stat_of(x):
    s = norm(x)
    if not s or len(s) > 40:
        return None
    if "mediana" in s:
        return "mediana"
    if "promedio" in s:
        return "promedio"
    if "desv" in s:
        return "desvio"
    if s.startswith("max"):
        return "max"
    if s.startswith("min"):
        return "min"
    m = re.search(r"percentil\s*(\d+)", s)
    if m:
        return f"p{m.group(1)}"
    if "cantidad" in s or "particip" in s:
        return "n"
    return None


def num(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        try:
            return float(str(x).replace(",", ".").strip())
        except ValueError:
            return float("nan")


def blank(v) -> bool:
    return pd.isna(v) or not str(v).strip()


def title_above(df, r) -> str:
    for rr in range(r - 1, max(r - 5, -1), -1):
        for c in range(df.shape[1]):
            v = df.iat[rr, c]
            if not blank(v) and stat_of(v) is None:
                return str(v).strip()[:200]
    return ""


def parse_sheet(df, sheet):
    """Devuelve registros crudos (uno por fila de cada bloque de la hoja)."""
    nR, nC = df.shape
    S = [[stat_of(df.iat[r, c]) for c in range(nC)] for r in range(nR)]
    hdr_rows = [r for r in range(nR) if sum(1 for c in range(nC) if S[r][c]) >= 3]
    hdr_set = set(hdr_rows)
    recs = []
    for r in hdr_rows:
        cols, seen = [], {}
        for c in range(nC):
            st = S[r][c]
            if st:
                seen[st] = seen.get(st, 0) + 1
                cols.append((c, st if seen[st] == 1 else f"{st}_{seen[st]}"))
        c0 = cols[0][0]
        ref_col = next((c for c in range(c0) if norm(df.iat[r, c]) == "referencia"), None)
        title = title_above(df, r)
        blanks = 0
        for rr in range(r + 1, nR):
            if rr in hdr_set:
                break
            vals = {name: num(df.iat[rr, c]) for c, name in cols}
            if all(pd.isna(v) for v in vals.values()):
                blanks += 1
                if blanks >= 2:
                    break
                continue
            blanks = 0
            per = next((df.iat[rr, c] for c in range(c0)
                        if c != ref_col and not blank(df.iat[rr, c])), None)
            ref = df.iat[rr, ref_col] if ref_col is not None else None
            recs.append({"hoja": sheet, "bloque": title, "periodo_raw": per,
                         "referencia": None if blank(ref) else str(ref).strip(), **vals})
    return recs


# ---- normalizacion ---------------------------------------------------------
def map_variable(title: str):
    """(codigo, mapeado_ok)"""
    t = norm(title)
    if t.startswith("precios minoristas"):
        return f"IPC_{'NUCLEO' if 'nucleo' in t else 'NG'}_{'GBA' if ('gba' in t or 'amba' in t) else 'NAC'}", True
    if t.startswith("tasa"):
        for key, code in (("badlar", "TASA_BADLAR"), ("tamar", "TASA_TAMAR"), ("leliq", "TASA_LELIQ"),
                          ("lebac", "TASA_LEBAC35"), ("pase", "TASA_PASE7")):
            if key in t:
                return code, True
        return "TASA_POLITICA", "politica monetaria" in t
    for key, code in (("tipo de cambio", "TC_NOMINAL"), ("resultado primario", "RESULTADO_PRIMARIO_SPNF"),
                      ("pib", "PIB"), ("exportaciones", "EXPORTACIONES"),
                      ("importaciones", "IMPORTACIONES"), ("desocupacion", "DESOCUPACION")):
        if key in t:
            return code, True
    return re.sub(r"[^a-z0-9]+", "_", t).strip("_").upper() or "SIN_TITULO", False


def split_ref(ref):
    """'TNA; %; dic-16' -> ('TNA; %', Timestamp dic-16)"""
    if not ref:
        return None, None
    m = re.search(r";\s*((?:[A-Za-z]{3,4}-\d{2})|(?:Trim\.\s*[IViv]+-\d{2}))\s*$", ref)
    if m:
        return ref[:m.start()].strip(), parse_date_txt(m.group(1))
    return ref, None


def classify_period(raw, ref_date):
    """-> (tipo, fecha_objetivo) o (None, None)"""
    if isinstance(raw, (pd.Timestamp, dt.datetime, dt.date)):
        return "mes", month_end(raw.year, raw.month)
    s = norm(raw)
    if re.fullmatch(r"\d{4}(\.0)?", s):
        return "anio", pd.Timestamp(year=int(float(s)), month=12, day=31)
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}.*", s):
        d = pd.to_datetime(s[:10])
        return "mes", month_end(d.year, d.month)
    if s.startswith("prox"):
        n = re.search(r"(\d+)", s)
        return (f"prox_{n.group(1)}m" if n else "prox"), ref_date
    d = parse_trim(s)
    if d is not None:
        return "trim", d
    d = parse_mmm_yy(s)
    if d is not None:
        return "mes", d
    return None, None


def relevamiento_from_sheet(df):
    for r in range(min(6, len(df))):
        for c in range(df.shape[1]):
            s = norm(df.iat[r, c])
            m = re.search(r"(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|setiembre|"
                          r"octubre|noviembre|diciembre)\s+(?:de\s+)?(\d{4})", s)
            if m:
                name = "septiembre" if m.group(1) == "setiembre" else m.group(1)
                return f"{int(m.group(2))}-{MESES_FULL.index(name) + 1:02d}"
    return None


def relevamiento_from_name(p: Path):
    n = p.name
    m = re.search(r"(\d{4})-(\d{2})", n)
    if m:
        return f"{m.group(1)}-{m.group(2)}"
    m = re.match(r"REM(\d{2})(\d{2})(\d{2})", n, flags=re.I)
    if m:
        return f"20{m.group(1)}-{m.group(2)}"
    m = re.search(r"-(" + "|".join(MESES) + r")[a-z]*-?(\d{4})", n.lower())
    if m:
        return f"{m.group(2)}-{MES_IDX[m.group(1)]:02d}"
    return None


def normalize(recs, rel):
    ry, rm = int(rel[:4]), int(rel[5:7])
    out = []
    for r in recs:
        unidad, ref_date = split_ref(r["referencia"])
        tipo, target = classify_period(r["periodo_raw"], ref_date)
        var, ok = map_variable(r["bloque"])
        r = dict(r)
        r.update({
            "relevamiento": rel, "variable": var, "variable_mapeada": ok,
            "unidad": unidad, "periodo_tipo": tipo,
            "fecha_objetivo": target,
            "horizonte_meses": None if target is None else (target.year - ry) * 12 + target.month - rm,
            "periodo_raw": "" if r["periodo_raw"] is None else str(r["periodo_raw"]),
        })
        out.append(r)
    return out


def parse_all(exclude_years):
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(list(RAW.glob("*.xlsx")) + list(RAW.glob("*.xls")))
    rows, report, seen_rel = [], [], {}
    for f in files:
        try:
            sheets = pd.read_excel(f, sheet_name=None, header=None)
        except Exception as e:  # noqa: BLE001
            report.append(f"\n{f.name}: ERROR al leer ({e})")
            continue
        main = [k for k in sheets if norm(k).startswith("cuadros")]
        if not main:
            report.append(f"\n{f.name}: sin hoja 'Cuadros de resultados' (hojas: {list(sheets)})")
            continue
        df0 = sheets[main[0]]
        rel_t, rel_n = relevamiento_from_sheet(df0), relevamiento_from_name(f)
        rel = rel_t or rel_n
        if rel is None:
            report.append(f"\n{f.name}: no pude determinar el mes de relevamiento")
            continue
        if int(rel[:4]) in exclude_years:
            continue
        head = f"\n=== {f.name} -> relevamiento {rel}"
        if rel_t and rel_n and rel_t != rel_n:
            head += f"  (AVISO: el nombre sugiere {rel_n})"
        report.append(head)
        if rel in seen_rel:
            report.append(f"  AVISO: {rel} ya venia de {seen_rel[rel]}; se usa este ultimo")
            rows = [x for x in rows if x["relevamiento"] != rel]
        seen_rel[rel] = f.name
        ignored = [k for k in sheets if k not in main]
        if ignored:
            report.append(f"  hojas ignoradas: {ignored}")
        recs = normalize(parse_sheet(df0, main[0]), rel)
        for var in dict.fromkeys(r["variable"] for r in recs):
            n = sum(1 for r in recs if r["variable"] == var)
            mapped = all(r["variable_mapeada"] for r in recs if r["variable"] == var)
            report.append(f"  {var:26s} {n:3d} filas{'' if mapped else '   <-- SIN MAPEAR'}")
        bad = [r for r in recs if r["periodo_tipo"] is None]
        if bad:
            report.append(f"  AVISO: {len(bad)} filas con periodo no reconocido, ej: "
                          f"{[r['periodo_raw'] for r in bad[:5]]}")
        nofecha = [r for r in recs if r["periodo_tipo"] and r["fecha_objetivo"] is None]
        if nofecha:
            report.append(f"  AVISO: {len(nofecha)} filas 'prox_*' sin fecha en Referencia")
        if not recs:
            report.append("  !! SIN REGISTROS")
        rows.extend(recs)
        print(f"{f.name}: {len(recs)} filas")

    (OUT / "structure_report.txt").write_text("\n".join(report), encoding="utf-8")
    if not rows:
        print("No se genero ninguna fila. Ver data/structure_report.txt")
        return
    df = pd.DataFrame(rows)
    lead = ["relevamiento", "variable", "unidad", "periodo_tipo", "periodo_raw", "fecha_objetivo",
            "horizonte_meses", "referencia", "bloque", "variable_mapeada", "hoja"]
    stats = [c for c in df.columns if c not in lead]
    df = df[lead + stats].sort_values(["relevamiento", "variable", "fecha_objetivo"])
    df.to_csv(OUT / "rem_long.csv", index=False, encoding="utf-8")
    try:
        df.to_parquet(OUT / "rem_long.parquet", index=False)
    except Exception:  # pyarrow no instalado: no pasa nada
        pass
    print(f"\nListo: {len(df)} filas en data/rem_long.csv. Reporte: data/structure_report.txt")


def inspect(path, rows):
    for sh, df in pd.read_excel(path, sheet_name=None, header=None).items():
        print(f"\n##### Hoja: {sh}  shape={df.shape}")
        print(df.head(rows).to_string(max_colwidth=28))


# ----------------------------------------------------------------------------
def main():
    global RAW, OUT
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    d = sub.add_parser("download")
    p = sub.add_parser("parse")
    a = sub.add_parser("all")
    for x in (d, a):
        x.add_argument("--start-year", type=int, default=2016)
    for x in (d, p, a):
        x.add_argument("--exclude-years", type=int, nargs="*", default=[2012])
    for x in (p, a):
        x.add_argument("--raw-dir")
        x.add_argument("--out-dir")
    i = sub.add_parser("inspect")
    i.add_argument("file")
    i.add_argument("--rows", type=int, default=40)
    args = ap.parse_args()

    cmd = args.cmd or "all"
    if cmd == "inspect":
        inspect(args.file, args.rows)
        return
    if getattr(args, "raw_dir", None):
        RAW = Path(args.raw_dir)
    if getattr(args, "out_dir", None):
        OUT = Path(args.out_dir)
    excl = getattr(args, "exclude_years", [2012])
    if cmd in ("download", "all"):
        download(getattr(args, "start_year", 2016), excl)
    if cmd in ("parse", "all"):
        parse_all(excl)


if __name__ == "__main__":
    main()
