"""convertir_ipc_bcr.py — Convierte el archivo ancho del IPC (BCR/ONEC) al CSV largo del proyecto.

Entrada: CSV/XLSX descargado de https://onec.bcr.gob.sv (fila de años, fila de meses, una fila por concepto).
Salida:  CSV con columnas concepto, Anio, Mes, ipc  (el que lee data_loading.cargar_ipc).

Uso:
  python scripts/extraccion/convertir_ipc_bcr.py --entrada ipc_alimentos.csv \
      --concepto "1.1 Alimentos" --salida data/external/ipc_alimentos.csv
"""
import argparse, csv, re
from pathlib import Path
import pandas as pd

MESES = {"ene": 1, "feb": 2, "mar": 3, "abr": 4, "may": 5, "jun": 6,
         "jul": 7, "ago": 8, "sep": 9, "oct": 10, "nov": 11, "dic": 12}


def leer_ipc_ancho(ruta, concepto):
    filas = list(csv.reader(open(ruta, encoding="utf-8-sig")))
    i_anio = next(i for i, f in enumerate(filas) if "concepto" in (f[0] or "").lower())
    anios, meses = filas[i_anio][1:], filas[i_anio + 1][1:]
    fila = next(f for f in filas[i_anio + 2:] if concepto.lower() in (f[0] or "").lower())
    out, anio = [], None
    for a, m, v in zip(anios, meses, fila[1:]):
        if a.strip():
            anio = int(a)
        m = m.strip().lower()[:3]
        if m in MESES and v.strip():
            out.append({"concepto": re.sub(r"\s+", " ", fila[0]).strip(), "Anio": anio,
                        "Mes": MESES[m], "ipc": float(v)})
    return pd.DataFrame(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--entrada", required=True)
    ap.add_argument("--concepto", default="1.1 Alimentos")
    ap.add_argument("--salida", default="data/external/ipc_alimentos.csv")
    a = ap.parse_args()
    d = leer_ipc_ancho(a.entrada, a.concepto)
    Path(a.salida).parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(a.salida, index=False)
    print(f"{len(d)} meses ({d.Anio.iloc[0]}-{d.Mes.iloc[0]:02d} a {d.Anio.iloc[-1]}-{d.Mes.iloc[-1]:02d}) -> {a.salida}")
