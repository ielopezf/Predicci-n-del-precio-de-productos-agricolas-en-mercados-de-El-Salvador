"""pronostico.py — Pronóstico a 1–4 semanas con intervalo empírico y semáforo de confiabilidad.

Qué hace (y qué NO hace):
- Pronóstico puntual = persistencia (último precio observado). En la evaluación de la etapa final
  ningún modelo supera a la persistencia de forma robusta (ver reports/etapa3), así que el modelo por
  defecto de la app es el que el informe respalda.
- Intervalo = cuantiles de la variación relativa histórica y_{t+h}/p_t - 1 por producto y horizonte,
  calibrados SOLO con objetivos anteriores a `FIN_CALIBRACION` (igual que la evaluación). En la prueba
  esos intervalos cubrieron menos que el nivel nominal (63 % de 80 %; 73 % de 90 %): son orientativos.
- Semáforo = error de la persistencia en el tramo de prueba para ese horizonte, contra la meta
  MAPE < 15 % de `config`. Es una medida descriptiva del pasado, no una garantía a futuro.
- Los datos terminan en junio de 2024: si el precio de partida es viejo, se avisa.
Funciones puras (sin Streamlit/Flask) para poder probarlas con pytest.
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

HORIZONTES = (1, 2, 3, 4)
NIVELES = (0.80, 0.90)
FIN_CALIBRACION = pd.Timestamp("2023-11-26")    # = INI_TEST de scripts/a4_horizontes.py
UMBRAL_VERDE, UMBRAL_AMARILLO = 5.0, config.META_MAPE_OBJETIVO * 100   # % de MAPE
DIAS_ALERTA_DATO_VIEJO = 21
MIN_OBS_CALIBRACION = 20


def _sin_tildes(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)).upper()


def cargar_historia(ruta: str | Path | None = None) -> pd.DataFrame:
    """Serie semanal por producto: columnas Fecha, Grupo, Producto, Precio (ordenada)."""
    d = pd.read_csv(ruta or config.DATASET_MODELADO, parse_dates=[config.COL_FECHA])
    d = d[[config.COL_FECHA, "Grupo", config.COL_PRODUCTO, config.COL_PRECIO]].dropna(subset=[config.COL_PRECIO])
    return d.sort_values(["Grupo", config.COL_FECHA]).reset_index(drop=True)


def variaciones_relativas(precios: pd.Series, h: int, fechas: pd.Series | None = None,
                          hasta: pd.Timestamp | None = None) -> np.ndarray:
    """y_{t+h}/p_t - 1; si se da `hasta`, solo objetivos con fecha < hasta (sin mirar el futuro)."""
    p = precios.to_numpy(float)
    if len(p) <= h:
        return np.array([])
    r = p[h:] / p[:-h] - 1
    if hasta is not None and fechas is not None:
        f_obj = pd.to_datetime(fechas.iloc[h:]).to_numpy()
        r = r[f_obj < np.datetime64(hasta)]
    return r[np.isfinite(r)]


def intervalo(precio_base: float, rel: np.ndarray, nivel: float) -> tuple[float, float]:
    if len(rel) < MIN_OBS_CALIBRACION:
        raise ValueError(f"Muy pocas observaciones para calibrar el intervalo ({len(rel)}).")
    a = (1 - nivel) / 2
    lo, hi = np.quantile(rel, [a, 1 - a])
    return max(precio_base * (1 + lo), 0.0), precio_base * (1 + hi)


def mape_persistencia_prueba(precios: pd.Series, fechas: pd.Series, h: int,
                             desde: pd.Timestamp = FIN_CALIBRACION) -> float:
    """MAPE (%) de predecir con el último precio, sobre objetivos >= `desde`."""
    p = precios.to_numpy(float)
    if len(p) <= h:
        return float("nan")
    f_obj = pd.to_datetime(fechas.iloc[h:]).to_numpy()
    m = f_obj >= np.datetime64(desde)
    if m.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(p[h:][m] - p[:-h][m]) / p[h:][m]) * 100)


def semaforo(mape: float) -> str:
    if not np.isfinite(mape):
        return "gris"
    return "verde" if mape < UMBRAL_VERDE else ("amarillo" if mape <= UMBRAL_AMARILLO else "rojo")


def avisos(grupo: str, fecha_objetivo: pd.Timestamp, fecha_base: pd.Timestamp,
           hoy: pd.Timestamp | None = None) -> list[str]:
    out = []
    if grupo == "TOMATE":
        out.append("Tomate: es el producto más volátil (variación media semanal ≈ 15 %). "
                   "El intervalo es muy ancho y el punto es poco informativo.")
    if fecha_objetivo.month in config.MESES_COSECHA.get(grupo, []):
        out.append("La fecha objetivo cae en meses de cosecha/temporada de este producto: "
                   "los precios suelen moverse más.")
    hoy = hoy if hoy is not None else pd.Timestamp.today().normalize()
    dias = (hoy - fecha_base).days
    if dias > DIAS_ALERTA_DATO_VIEJO:
        out.append(f"El último precio disponible tiene {dias} días de antigüedad. "
                   "Cargue el precio vigente del reporte del MAG para un pronóstico útil.")
    return out


@dataclass
class Pronostico:
    producto: str
    grupo: str
    fecha_base: pd.Timestamp
    precio_base: float
    filas: list = field(default_factory=list)

    def tabla(self) -> pd.DataFrame:
        return pd.DataFrame(self.filas)


def pronosticar(historia: pd.DataFrame, grupo: str, precio_base: float | None = None,
                fecha_base: pd.Timestamp | None = None, horizontes=HORIZONTES,
                hoy: pd.Timestamp | None = None) -> Pronostico:
    """Pronóstico por persistencia para un producto. Si no se da precio_base, usa el último de la historia."""
    h_ = historia[historia["Grupo"] == grupo].sort_values(config.COL_FECHA)
    if h_.empty:
        raise ValueError(f"Producto desconocido: {grupo}")
    if precio_base is not None and not precio_base > 0:
        raise ValueError("El precio base debe ser positivo.")
    p_hist, f_hist = h_[config.COL_PRECIO].reset_index(drop=True), h_[config.COL_FECHA].reset_index(drop=True)
    pb = float(precio_base) if precio_base is not None else float(p_hist.iloc[-1])
    fb = pd.Timestamp(fecha_base) if fecha_base is not None else pd.Timestamp(f_hist.iloc[-1])
    res = Pronostico(h_["Producto"].iloc[0], grupo, fb, pb)
    for h in horizontes:
        rel = variaciones_relativas(p_hist, h, f_hist, FIN_CALIBRACION)
        f_obj = fb + pd.Timedelta(weeks=h)
        mape = mape_persistencia_prueba(p_hist, f_hist, h)
        fila = {"producto": res.producto, "grupo": grupo, "horizonte_sem": h, "fecha_base": fb.date(),
                "fecha_objetivo": f_obj.date(), "precio_base": round(pb, 2), "pronostico": round(pb, 2),
                "mape_prueba_%": round(mape, 2), "semaforo": semaforo(mape),
                "avisos": " | ".join(avisos(grupo, f_obj, fb, hoy))}
        for n in NIVELES:
            lo, hi = intervalo(pb, rel, n)
            fila[f"lim_inf_{int(n*100)}"], fila[f"lim_sup_{int(n*100)}"] = round(lo, 2), round(hi, 2)
        res.filas.append(fila)
    return res


def pronosticar_todos(historia: pd.DataFrame, precios_actuales: dict | None = None,
                      fecha_base: pd.Timestamp | None = None, hoy: pd.Timestamp | None = None) -> pd.DataFrame:
    precios_actuales = precios_actuales or {}
    partes = [pronosticar(historia, g, precios_actuales.get(g), fecha_base, hoy=hoy).tabla()
              for g in sorted(historia["Grupo"].unique())]
    return pd.concat(partes, ignore_index=True)


def explicacion(fila: pd.Series) -> str:
    """Texto en lenguaje llano para una fila de pronóstico."""
    color = {"verde": "alta", "amarillo": "media", "rojo": "baja", "gris": "no evaluable"}[fila["semaforo"]]
    return (f"Se espera que el precio de {fila['producto'].title()} en {int(fila['horizonte_sem'])} semana(s) "
            f"siga cerca del actual (${fila['precio_base']:.2f}). Con 80 % de confianza histórica estaría entre "
            f"${fila['lim_inf_80']:.2f} y ${fila['lim_sup_80']:.2f}. Confiabilidad {color}: en el período de prueba "
            f"este método se equivocó en promedio {fila['mape_prueba_%']:.1f} % a este plazo.")


def leer_reporte_mag(archivo) -> dict:
    """Lee un CSV/XLSX simple con columnas Fecha, Producto, Precio y devuelve {grupo: (fecha, precio)}
    con la última fila de cada producto reconocido (por nombre: ARROZ, FRIJOL, MAÍZ, NARANJA, TOMATE)."""
    nombre = str(getattr(archivo, "name", archivo)).lower()
    df = pd.read_excel(archivo) if nombre.endswith(("xlsx", "xls")) else pd.read_csv(archivo)
    df.columns = [str(c).strip().lower() for c in df.columns]
    faltan = {"fecha", "producto", "precio"} - set(df.columns)
    if faltan:
        raise ValueError(f"Faltan columnas: {sorted(faltan)}. Se requieren Fecha, Producto, Precio.")
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df["precio"] = pd.to_numeric(df["precio"], errors="coerce")
    df = df.dropna(subset=["fecha", "precio"]).sort_values("fecha")
    out = {}
    for g in ("ARROZ", "FRIJOL", "MAÍZ", "NARANJA", "TOMATE"):
        clave = _sin_tildes(g)
        sub = df[df["producto"].astype(str).map(_sin_tildes).str.contains(clave)]
        if not sub.empty and sub["precio"].iloc[-1] > 0:
            out[g] = (sub["fecha"].iloc[-1], float(sub["precio"].iloc[-1]))
    return out
