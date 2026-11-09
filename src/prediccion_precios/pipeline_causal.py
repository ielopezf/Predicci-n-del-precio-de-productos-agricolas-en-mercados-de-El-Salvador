"""pipeline_causal.py — Dataset semanal CAUSAL (tarea A3).

Diferencias con el pipeline original (preprocessing.py):
  * sin recorte IQR global (los picos son lo que se quiere anticipar);
  * el precio semanal NO se interpola: la semana sin dato queda NaN y se marca
    `precio_imputado`; el precio vigente (`precio_vigente`) usa solo ffill (pasado);
  * sin bfill ni interpolación bidireccional (no usa valores futuros);
  * variables nuevas: `n_dias` (días con cotización en la semana) y
    `semanas_sin_cambio` (semanas seguidas con el mismo precio vigente).
Las variables externas (combustible, lluvia, IPC, producción) se toman del dataset
ya integrado: son exógenas y no dependen del precio.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, auditoria_datos as au

EXTERNAS = ["diesel", "gas_regular", "gas_especial", "precipitacion_mm", "lluvia_max_mm",
            "ipc_transporte", "area_ha", "produccion_t", "rendimiento_kg_ha"]


def semanal_causal(ruta_precios):
    raw = au.cargar_precios_crudos(ruta_precios).dropna(subset=[config.COL_PRECIO])
    d = raw.groupby(["Grupo", config.COL_FECHA])[config.COL_PRECIO].agg(["mean", "size"])
    partes = []
    for g, x in d.groupby(level=0):
        x = x.droplevel(0)
        w = x["mean"].resample(config.FRECUENCIA).mean()
        n = x["size"].resample(config.FRECUENCIA).sum().reindex(w.index).fillna(0).astype(int)
        t = pd.DataFrame({config.COL_FECHA: w.index, "Grupo": g, "precio_obs": w.to_numpy(), "n_dias": n.to_numpy()})
        t["precio_imputado"] = t["precio_obs"].isna().astype(int)
        t["precio_vigente"] = t["precio_obs"].ffill()
        same = (t["precio_vigente"].diff().abs() < 1e-9)
        grp = (~same).cumsum()
        t["semanas_sin_cambio"] = same.groupby(grp).cumsum()
        partes.append(t)
    return pd.concat(partes, ignore_index=True)


def con_externas(causal, dataset_modelado):
    ext = dataset_modelado[[config.COL_FECHA, "Grupo"] + EXTERNAS]
    return causal.merge(ext, on=[config.COL_FECHA, "Grupo"], how="inner")
