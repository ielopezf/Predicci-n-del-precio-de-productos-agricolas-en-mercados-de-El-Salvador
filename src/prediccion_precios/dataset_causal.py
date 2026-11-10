"""dataset_causal.py — Dataset de modelado CAUSAL para reentrenar los modelos de la Etapa 2.

Misma estructura de columnas que `dataset_modelado.csv` (así `features.construir_matriz_modelado`,
los modelos y la API funcionan sin cambios), pero con el tratamiento de la etapa final:
  * sin recorte IQR global;
  * sin interpolar ni bfill: la semana sin cotización conserva el último precio vigente (ffill);
  * lags y medias móviles se recalculan con ese precio.
`precio_imputado`, `n_dias` y `semanas_sin_cambio` se guardan aparte (FLAGS_CAUSAL) para no
alterar el conjunto de variables de los modelos.
"""
from __future__ import annotations

import pandas as pd

from . import config, features as ft, pipeline_causal as pc


def armar_dataset_causal(causal_ext: pd.DataFrame, producto_por_grupo: dict, columnas: list):
    """causal_ext: salida de `pipeline_causal.con_externas`. columnas: orden de columnas del dataset de la Etapa 2.

    Devuelve (dataset, flags). Se descartan las semanas iniciales sin ningún precio previo.
    """
    d = causal_ext.sort_values(["Grupo", config.COL_FECHA]).reset_index(drop=True)
    d = d[d["precio_vigente"].notna()].copy()
    d[config.COL_PRECIO] = d["precio_vigente"]
    d[config.COL_PRODUCTO] = d["Grupo"].map(producto_por_grupo)
    f = pd.to_datetime(d[config.COL_FECHA])
    d["Anio"], d["Mes"] = f.dt.year, f.dt.month
    d = ft.agregar_variables_calendario(d)
    d = ft.agregar_lags(d)
    d = ft.agregar_medias_moviles(d)
    d = ft.marcar_temporada_cosecha(d)
    flags = d[[config.COL_FECHA, "Grupo", "precio_imputado", "n_dias", "semanas_sin_cambio"]].copy()
    return d[columnas].reset_index(drop=True), flags.reset_index(drop=True)


def construir(ruta_precios, ruta_etapa2=None):
    """Lee los precios crudos y el dataset de la Etapa 2 (solo para columnas externas) y arma el causal."""
    base = pd.read_csv(ruta_etapa2 or config.DATASET_MODELADO, parse_dates=[config.COL_FECHA])
    causal = pc.semanal_causal(ruta_precios)
    ext = pc.con_externas(causal, base)
    producto = base.drop_duplicates("Grupo").set_index("Grupo")[config.COL_PRODUCTO].to_dict()
    return armar_dataset_causal(ext, producto, list(base.columns))
