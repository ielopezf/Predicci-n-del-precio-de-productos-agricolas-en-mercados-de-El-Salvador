"""Pruebas mínimas de A1: la persistencia debe coincidir con el rezago 1 del dataset."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from prediccion_precios import baselines_ingenuos as bl, config, evaluation as ev  # noqa: E402


def _datos():
    return pd.read_csv(config.DATASET_MODELADO, parse_dates=[config.COL_FECHA])


def test_persistencia_h1_coincide_con_precio_lag1():
    df = _datos()
    det = bl.evaluar(df, metodos=("persistencia",), horizontes=(1,))
    d = df.sort_values(config.COL_FECHA).reset_index(drop=True).dropna().reset_index(drop=True)
    prueba = d.iloc[-157:]
    mape_lag1 = np.mean(np.abs(prueba.Precio - prueba.precio_lag1) / prueba.Precio) * 100
    mape_mod = np.mean(np.abs(det.real - det.predicho) / det.real) * 100
    assert len(det) == 157
    assert abs(mape_lag1 - mape_mod) < 1e-9


def test_mase_de_persistencia_en_serie_constante_es_cero():
    assert ev.mase([5, 5, 5], [5, 5, 5], escala=1.0) == 0.0


def test_pronosticos_tienen_h_max_valores():
    hist = np.linspace(10, 20, 80)
    for m in bl.METODOS:
        assert len(bl.pronosticar(m, hist, 8)) == 8
