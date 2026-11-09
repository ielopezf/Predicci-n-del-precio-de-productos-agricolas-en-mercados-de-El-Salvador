"""A2 — Auditoría de calidad de datos (nulos, repeticiones, recortes y cobertura de fuentes).

Uso:
    python scripts/a2_auditoria.py --precios data/raw/precios_consolidados.xlsx \
        --combustibles data/external/precios_combustibles_sv.xlsx \
        --precip-diaria data/external/precipitaciones_el_salvador_2021-2024.xlsx \
        --lluvia-mensual data/external/lluvia_mensual_sv.xlsx \
        --faostat data/external/FAOSTAT_data_en_7-13-2026.csv

Salida en reports/etapa3/: auditoria_precios.csv, semanas_sin_dato.csv, cobertura_fuentes.csv
"""
import argparse
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from prediccion_precios import auditoria_datos as au  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--precios", required=True)
    ap.add_argument("--combustibles")
    ap.add_argument("--precip-diaria")
    ap.add_argument("--lluvia-mensual")
    ap.add_argument("--faostat")
    a = ap.parse_args()

    raw = au.cargar_precios_crudos(a.precios)
    sal = RAIZ / "reports" / "etapa3"
    sal.mkdir(parents=True, exist_ok=True)

    aud = au.auditar_precios(raw)
    aud.to_csv(sal / "auditoria_precios.csv", index=False)
    au.semanas_sin_dato(raw).to_csv(sal / "semanas_sin_dato.csv", index=False)

    rutas = {"precios": a.precios}
    for k, v in (("combustibles", a.combustibles), ("precip_diaria", a.precip_diaria),
                 ("lluvia_mensual", a.lluvia_mensual), ("faostat", a.faostat)):
        if v:
            rutas[k] = v
    cob = au.cobertura_fuentes(rutas)
    cob.to_csv(sal / "cobertura_fuentes.csv", index=False)

    import pandas as pd
    pd.set_option("display.width", 200, "display.max_columns", 30)
    print(aud.T.to_string())
    print()
    print(cob.to_string(index=False))


if __name__ == "__main__":
    main()
