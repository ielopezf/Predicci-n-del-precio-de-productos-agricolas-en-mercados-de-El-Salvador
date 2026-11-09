"""
Extractor de precipitación diaria - SNET/MARN El Salvador

Fuente:
https://mapas.snet.gob.sv/hidrologia/lluvia.php?fecha=AAAA-M-D

Genera un consolidado diario por estación para el período 2020-2026.

Salidas:
- precipitaciones_el_salvador_2020_2026.xlsx
- precipitaciones_el_salvador_2020_2026.csv
- precipitaciones_avance.csv
- precipitaciones_extraccion.log

Requisitos:
    pip install requests beautifulsoup4 pandas openpyxl lxml
"""

from __future__ import annotations

import logging
import random
import re
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

URL_BASE = "https://mapas.snet.gob.sv/hidrologia/lluvia.php"
FECHA_INICIO = date(2020, 1, 1)
FECHA_FIN = min(date.today(), date(2026, 12, 31))

ARCHIVO_EXCEL = Path("precipitaciones_el_salvador_2020_2026.xlsx")
ARCHIVO_CSV = Path("precipitaciones_el_salvador_2020_2026.csv")
ARCHIVO_AVANCE = Path("precipitaciones_avance.csv")
ARCHIVO_LOG = Path("precipitaciones_extraccion.log")

ESPERA_MIN_SEG = 0.35
ESPERA_MAX_SEG = 0.80
GUARDAR_CADA = 30
TIMEOUT_SEG = 30
REANUDAR = True

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(ARCHIVO_LOG, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def crear_sesion() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 Chrome/124.0 Safari/537.36 "
            "Proyecto-academico-UES/1.0"
        ),
        "Accept-Language": "es-SV,es;q=0.9,en;q=0.7",
    })
    return session


def rango_fechas(inicio: date, fin: date):
    actual = inicio
    while actual <= fin:
        yield actual
        actual += timedelta(days=1)


def texto_limpio(valor: object) -> str:
    return re.sub(r"\s+", " ", str(valor or "")).strip()


def a_float(valor: object) -> Optional[float]:
    txt = texto_limpio(valor)
    if not txt or txt in {"-", "--", "N/D", "ND", "null", "None"}:
        return None
    txt = txt.replace(",", ".")
    match = re.search(r"-?\d+(?:\.\d+)?", txt)
    if not match:
        return None
    try:
        return float(match.group())
    except ValueError:
        return None


def extraer_id_estacion(celda) -> Optional[int]:
    enlace = celda.find("a", href=True)
    candidatos = []
    if enlace:
        candidatos.extend([enlace.get_text(" ", strip=True), enlace.get("href", "")])
    candidatos.append(celda.get_text(" ", strip=True))
    for candidato in candidatos:
        numeros = re.findall(r"\d+", candidato)
        if numeros:
            try:
                return int(numeros[-1])
            except ValueError:
                pass
    return None


def descargar_html(session: requests.Session, fecha: date) -> str:
    fecha_param = f"{fecha.year}-{fecha.month}-{fecha.day}"
    respuesta = session.get(
        URL_BASE,
        params={"fecha": fecha_param},
        timeout=TIMEOUT_SEG,
    )
    respuesta.raise_for_status()
    if not respuesta.encoding or respuesta.encoding.lower() == "iso-8859-1":
        respuesta.encoding = respuesta.apparent_encoding or "utf-8"
    return respuesta.text


def localizar_tabla(soup: BeautifulSoup):
    for tabla in soup.find_all("table"):
        encabezado = texto_limpio(tabla.get_text(" ", strip=True)).lower()
        if "estacion" in encabezado and "acumulado" in encabezado:
            return tabla
    return None


def parsear_fecha(html: str, fecha_consulta: date) -> list[dict]:
    soup = BeautifulSoup(html, "lxml")
    tabla = localizar_tabla(soup)
    if tabla is None:
        tablas = soup.find_all("table")
        if tablas:
            tabla = max(tablas, key=lambda t: len(t.find_all("tr")))
        else:
            return []

    registros = []
    for fila in tabla.find_all("tr"):
        celdas = fila.find_all(["td", "th"])
        valores = [texto_limpio(c.get_text(" ", strip=True)) for c in celdas]
        if len(celdas) < 7:
            continue
        unido = " ".join(valores).lower()
        if "estacion" in unido or "promedio" in unido or "maximo" in unido:
            continue

        id_estacion = extraer_id_estacion(celdas[0])
        estacion = valores[1]
        if not estacion:
            continue

        hora_inicio = valores[2]
        valor_inicio = a_float(valores[3])
        hora_actual = valores[4]
        valor_actual = a_float(valores[5])
        acumulado = a_float(valores[6])

        if acumulado is None and valor_inicio is None and valor_actual is None:
            continue

        registros.append({
            "Fecha": fecha_consulta.isoformat(),
            "Año": fecha_consulta.year,
            "Mes": fecha_consulta.month,
            "Día": fecha_consulta.day,
            "ID Estación": id_estacion,
            "Estación": estacion,
            "Hora Inicio": hora_inicio or None,
            "Valor Inicio": valor_inicio,
            "Hora Actual": hora_actual or None,
            "Valor Actual": valor_actual,
            "Precipitación 24h (mm)": acumulado,
            "URL Fuente": (
                f"{URL_BASE}?fecha="
                f"{fecha_consulta.year}-{fecha_consulta.month}-{fecha_consulta.day}"
            ),
        })
    return registros


def cargar_avance() -> pd.DataFrame:
    if REANUDAR and ARCHIVO_AVANCE.exists():
        df = pd.read_csv(ARCHIVO_AVANCE, dtype={"ID Estación": "Int64"})
        logger.info("Avance encontrado: %s registros.", len(df))
        return df
    return pd.DataFrame()


def guardar_avance(registros: list[dict]) -> None:
    if registros:
        pd.DataFrame(registros).to_csv(
            ARCHIVO_AVANCE, index=False, encoding="utf-8-sig"
        )


def crear_resumen_diario(df: pd.DataFrame) -> pd.DataFrame:
    col = "Precipitación 24h (mm)"
    resumen = (
        df.groupby(["Fecha", "Año", "Mes", "Día"], as_index=False)
        .agg(
            Estaciones_reportadas=("Estación", "nunique"),
            Estaciones_con_lluvia=(col, lambda s: int((s.fillna(0) > 0).sum())),
            Precipitación_promedio_mm=(col, "mean"),
            Precipitación_mediana_mm=(col, "median"),
            Precipitación_máxima_mm=(col, "max"),
            Precipitación_total_estaciones_mm=(col, "sum"),
        )
    )
    for c in [
        "Precipitación_promedio_mm",
        "Precipitación_mediana_mm",
        "Precipitación_máxima_mm",
        "Precipitación_total_estaciones_mm",
    ]:
        resumen[c] = resumen[c].round(3)
    return resumen


def guardar_resultados(registros: list[dict], errores: list[dict]) -> None:
    df = pd.DataFrame(registros)
    if df.empty:
        logger.warning("No hay registros para guardar.")
        return

    df["Fecha"] = pd.to_datetime(df["Fecha"])
    df["ID Estación"] = pd.to_numeric(df["ID Estación"], errors="coerce").astype("Int64")
    df.sort_values(["Fecha", "Estación", "ID Estación"], inplace=True, ignore_index=True)

    resumen = crear_resumen_diario(df)
    df_errores = pd.DataFrame(errores, columns=["Fecha", "Tipo", "Detalle"])

    df.to_csv(ARCHIVO_CSV, index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(ARCHIVO_EXCEL, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Detalle", index=False)
        resumen.to_excel(writer, sheet_name="Resumen_diario", index=False)
        df_errores.to_excel(writer, sheet_name="Errores", index=False)

        for nombre_hoja in ["Detalle", "Resumen_diario", "Errores"]:
            hoja = writer.book[nombre_hoja]
            hoja.freeze_panes = "A2"
            hoja.auto_filter.ref = hoja.dimensions

    logger.info("Excel guardado en: %s", ARCHIVO_EXCEL.resolve())
    logger.info("CSV guardado en: %s", ARCHIVO_CSV.resolve())


def main() -> None:
    logger.info("=" * 72)
    logger.info("Extractor de precipitación diaria SNET - El Salvador")
    logger.info("Período: %s a %s", FECHA_INICIO, FECHA_FIN)
    logger.info("=" * 72)

    session = crear_sesion()
    df_avance = cargar_avance()

    registros: list[dict] = []
    errores: list[dict] = []
    fechas_completadas: set[str] = set()

    if not df_avance.empty:
        registros = df_avance.to_dict("records")
        fechas_completadas = set(pd.to_datetime(df_avance["Fecha"]).dt.date.astype(str))

    fechas = list(rango_fechas(FECHA_INICIO, FECHA_FIN))
    pendientes = [f for f in fechas if f.isoformat() not in fechas_completadas]

    logger.info(
        "Días totales: %s | Ya procesados: %s | Pendientes: %s",
        len(fechas), len(fechas_completadas), len(pendientes)
    )

    for indice, fecha_actual in enumerate(pendientes, start=1):
        try:
            html = descargar_html(session, fecha_actual)
            filas = parsear_fecha(html, fecha_actual)
            if filas:
                registros.extend(filas)
                logger.info(
                    "[%s/%s] %s: %s estaciones.",
                    indice, len(pendientes), fecha_actual, len(filas)
                )
            else:
                errores.append({
                    "Fecha": fecha_actual.isoformat(),
                    "Tipo": "Sin datos",
                    "Detalle": "No se encontró una tabla válida o no había estaciones.",
                })
                logger.warning("[%s/%s] %s: sin datos.", indice, len(pendientes), fecha_actual)
        except requests.RequestException as exc:
            errores.append({
                "Fecha": fecha_actual.isoformat(),
                "Tipo": "HTTP",
                "Detalle": str(exc),
            })
            logger.error("[%s/%s] %s: error HTTP: %s", indice, len(pendientes), fecha_actual, exc)
        except Exception as exc:
            errores.append({
                "Fecha": fecha_actual.isoformat(),
                "Tipo": "Parseo",
                "Detalle": repr(exc),
            })
            logger.exception("[%s/%s] %s: error inesperado.", indice, len(pendientes), fecha_actual)

        if indice % GUARDAR_CADA == 0:
            guardar_avance(registros)
            logger.info("Checkpoint guardado (%s registros).", len(registros))

        time.sleep(random.uniform(ESPERA_MIN_SEG, ESPERA_MAX_SEG))

    guardar_avance(registros)
    guardar_resultados(registros, errores)

    logger.info("=" * 72)
    logger.info("Registros obtenidos: %s", len(registros))
    logger.info("Fechas con incidencia: %s", len(errores))
    logger.info("=" * 72)


if __name__ == "__main__":
    main()
