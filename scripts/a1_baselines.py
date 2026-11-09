"""A1 — Baselines ingenuos y clásicos sobre el MISMO tramo de prueba (157 filas).

Uso (desde la raíz del repositorio):
    python scripts/a1_baselines.py

Salida: reports/etapa3/metricas_baselines.csv y reports/etapa3/predicciones_baselines.csv
"""
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from prediccion_precios import baselines_ingenuos as bl, config  # noqa: E402


def main():
    df = pd.read_csv(config.DATASET_MODELADO, parse_dates=[config.COL_FECHA])
    detalle = bl.evaluar(df, verbose=True)
    metricas = bl.resumir(detalle)

    salida = RAIZ / "reports" / "etapa3"
    salida.mkdir(parents=True, exist_ok=True)
    metricas.round(4).to_csv(salida / "metricas_baselines.csv", index=False)
    detalle.to_csv(salida / "predicciones_baselines.csv", index=False)

    g = metricas[metricas["producto"] == "GLOBAL"]
    print("\nMAPE % global por método y horizonte (semanas)")
    print(g.pivot(index="metodo", columns="horizonte", values="MAPE_%").round(2))
    print("\nMASE global (escala = error ingenuo a 1 semana en el periodo previo a la prueba;")
    print("la prueba es más volátil que el entrenamiento, por eso la persistencia da > 1)")
    print(g.pivot(index="metodo", columns="horizonte", values="MASE").round(2))
    print("\nRazón MAE frente a persistencia (< 1 = mejora a la persistencia)")
    print(g.pivot(index="metodo", columns="horizonte", values="razon_MAE_vs_persistencia").round(2))
    print("\nMAPE % a 1 semana por producto")
    p1 = metricas[(metricas["horizonte"] == 1) & (metricas["producto"] != "GLOBAL")]
    print(p1.pivot(index="metodo", columns="producto", values="MAPE_%").round(2))


if __name__ == "__main__":
    main()
