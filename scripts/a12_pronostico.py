"""a12_pronostico.py — Exporta el pronóstico a 1–4 semanas de los 5 productos.

Uso: python scripts/a12_pronostico.py [--mag reporte_mag.csv]
Salida: reports/pronostico.csv (la app y la API lo leen o lo recalculan con el mismo módulo).
--mag: CSV/XLSX con columnas Fecha, Producto, Precio para usar el último precio vigente de cada producto.
"""
import argparse, sys
from pathlib import Path
import pandas as pd
RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
from prediccion_precios import pronostico as pr  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--mag")
a = ap.parse_args()
h = pr.cargar_historia()
if a.mag:
    ult = pr.leer_reporte_mag(a.mag)
    precios = {g: p for g, (f, p) in ult.items()}
    fb = max(f for f, _ in ult.values())
    t = pr.pronosticar_todos(h, precios, fb)
else:
    t = pr.pronosticar_todos(h)
out = RAIZ / "reports" / "pronostico.csv"; out.parent.mkdir(exist_ok=True)
t.to_csv(out, index=False)
print(t[["grupo", "horizonte_sem", "fecha_objetivo", "pronostico", "lim_inf_80", "lim_sup_80", "mape_prueba_%", "semaforo"]].to_string(index=False))
