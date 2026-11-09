"""auditoria_datos.py — Auditoría de calidad de datos ANTES de imputar (tarea A2).

Responde, con números, a: ¿cuánto falta?, ¿dónde?, ¿qué tanto se repiten los
precios?, ¿qué recorta el filtro de outliers? y ¿hasta cuándo llega cada fuente?
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

# Variantes objetivo normalizadas (espacios simples) -> grupo
def _norm(s):
    return " ".join(str(s).split())


def cargar_precios_crudos(ruta_xlsx, solo_objetivo=True):
    """Lee precios_consolidados.xlsx: Precio numérico (n.d. -> NaN) y Grupo."""
    raw = pd.read_excel(ruta_xlsx)
    raw[config.COL_PRODUCTO] = raw[config.COL_PRODUCTO].map(_norm)
    raw[config.COL_PRECIO] = pd.to_numeric(raw[config.COL_PRECIO], errors="coerce")
    raw[config.COL_FECHA] = pd.to_datetime(raw[config.COL_FECHA])
    raw["Grupo"] = raw[config.COL_PRODUCTO].str.split().str[0]
    if solo_objetivo:
        objetivo = {_norm(p) for p in config.PRODUCTOS_OBJETIVO}
        raw = raw[raw[config.COL_PRODUCTO].isin(objetivo)]
    return raw.reset_index(drop=True)


def auditar_precios(raw):
    """Una fila por producto con los indicadores de calidad que importan para imputar."""
    filas = []
    for grupo, s in raw.groupby("Grupo"):
        s = s.sort_values(config.COL_FECHA)
        diario = s.groupby(config.COL_FECHA)[config.COL_PRECIO].mean()
        con_dato = diario.dropna()
        iguales = (con_dato.diff().abs() < 1e-9)
        sem = con_dato.index.to_period("W-SUN")
        dias_sem = con_dato.groupby(sem).size()
        q1, q3 = con_dato.quantile([.25, .75])
        lim_sup, lim_inf = q3 + 1.5 * (q3 - q1), q1 - 1.5 * (q3 - q1)
        filas.append({
            "producto": grupo,
            "registros": len(s),
            "registros_nd": int(s[config.COL_PRECIO].isna().sum()),
            "dias_con_dato": len(con_dato),
            "primer_dia": con_dato.index.min().date(),
            "ultimo_dia": con_dato.index.max().date(),
            "semanas_con_dato": int(dias_sem.size),
            "semanas_con_1_o_2_dias": int((dias_sem <= 2).sum()),
            "pct_dias_precio_identico_al_anterior": round(float(iguales.mean() * 100), 1),
            "pct_identico_2021": round(float(iguales[iguales.index.year == 2021].mean() * 100), 1),
            "pct_identico_2022": round(float(iguales[iguales.index.year == 2022].mean() * 100), 1),
            "pct_identico_2023": round(float(iguales[iguales.index.year == 2023].mean() * 100), 1),
            "pct_identico_2024": round(float(iguales[iguales.index.year == 2024].mean() * 100), 1),
            "limite_IQR_inferior": round(float(lim_inf), 2),
            "limite_IQR_superior": round(float(lim_sup), 2),
            "dias_recortados_IQR": int(((con_dato > lim_sup) | (con_dato < lim_inf)).sum()),
            "precio_max_crudo": round(float(con_dato.max()), 2),
        })
    return pd.DataFrame(filas)


def semanas_sin_dato(raw):
    """Semanas (domingo de cierre) sin ningún precio, por producto."""
    fil = []
    for grupo, s in raw.groupby("Grupo"):
        diario = s.dropna(subset=[config.COL_PRECIO])
        semanal = (diario.set_index(config.COL_FECHA)[config.COL_PRECIO]
                   .resample(config.FRECUENCIA).mean())
        for f in semanal[semanal.isna()].index:
            fil.append({"producto": grupo, "semana_cierre": f.date()})
    return pd.DataFrame(fil)


def cobertura_fuentes(rutas):
    """Rango real de cada fuente. `rutas` = dict nombre -> (ruta, lector) o (ruta, None)."""
    out = []

    def add(nombre, ini, fin, frec, nota=""):
        out.append({"fuente": nombre, "inicio": ini, "fin": fin, "frecuencia": frec, "nota": nota})

    if "precios" in rutas:
        r = cargar_precios_crudos(rutas["precios"], solo_objetivo=False)
        v = r.dropna(subset=[config.COL_PRECIO])[config.COL_FECHA]
        add("Precios MAG (consolidado)", v.min().date(), v.max().date(), "diaria (lun-vie)",
            "variable objetivo; limita todo el proyecto")
    if "combustibles" in rutas:
        c = pd.read_excel(rutas["combustibles"])
        f = pd.to_datetime(dict(year=c["Año"], month=c["Mes"], day=c["Día"]))
        add("Combustibles", f.min().date(), f.max().date(), "diaria",
            "'Diesel' regular = 0 en 2 672 filas (se sustituyó por bajo en azufre)")
    if "precip_diaria" in rutas:
        p = pd.read_excel(rutas["precip_diaria"])
        add("Precipitación diaria", p["Fecha"].min().date(), p["Fecha"].max().date(), "diaria",
            "promedio de 83-100 estaciones por día")
    if "lluvia_mensual" in rutas:
        m = pd.read_excel(rutas["lluvia_mensual"])
        m = m[m["Fecha"].astype(str).str.contains(r"\d{4}$")]
        add("Lluvia mensual", m["Fecha"].iloc[0], m["Fecha"].iloc[-1], "mensual")
    if "faostat" in rutas:
        f = pd.read_csv(rutas["faostat"])
        add("FAOSTAT producción", int(f["Year"].min()), int(f["Year"].max()), "anual",
            "2023-2024: valores imputados/estimados/no oficiales (flags I, E, X)")
    return pd.DataFrame(out)
