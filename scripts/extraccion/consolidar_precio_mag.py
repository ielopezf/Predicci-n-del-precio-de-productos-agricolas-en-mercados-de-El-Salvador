"""
Consolida el precio promedio de UN producto a partir de los Excel diarios del MAG (El Salvador).

- Fecha  -> celda B3 (datetime) de la hoja de Granos Básicos (o la hoja activa)
- Precio -> celda indicada con --celda-precio (por defecto D15, la que usaba el script original
            para MAÍZ BLANCO; el docstring original decía D9 pero el código leía D15)

IMPORTANTE: lee celdas fijas. Si el MAG cambia el orden de las filas, leerá otro producto sin avisar.
Por eso se puede pasar --texto-fila: si en las columnas A-C de esa fila no aparece ese texto, avisa.

Uso:
    python scripts/extraccion/consolidar_precio_mag.py excels_mag tabla_maiz_blanco.xlsx \
        --producto "MAÍZ BLANCO" --celda-precio D15 --texto-fila MAIZ
    (para otro producto, cambie --producto, --celda-precio y --texto-fila; la hoja puede variar:
     GranosBasicos, Hortalizas, Frutas -> use --hoja)

Resultado: Excel con columnas Producto | Día | Mes | Año | Precio

Requisitos:
    pip install openpyxl pandas
"""

import argparse
import os
import re
import unicodedata
import pandas as pd
from datetime import datetime
from openpyxl import load_workbook

# ──────────────────────────────────────────────
# CONFIGURACIÓN — ajusta estas rutas
# ──────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("carpeta", help="Carpeta con los .xlsx del MAG (salida de descargar_mag.py)")
ap.add_argument("salida", help="Excel de resultado")
ap.add_argument("--producto", default="MAÍZ BLANCO")
ap.add_argument("--celda-precio", default="D15")
ap.add_argument("--celda-fecha", default="B3")
ap.add_argument("--hoja", default=None, help="Texto parcial del nombre de la hoja (por defecto, la de granos básicos)")
ap.add_argument("--texto-fila", default=None, help="Texto que debe aparecer en las columnas A-C de la fila del precio")
_a = ap.parse_args()
CARPETA_EXCEL, ARCHIVO_SALIDA = _a.carpeta, _a.salida
PRODUCTO, CELDA_PRECIO, CELDA_FECHA, HOJA, TEXTO_FILA = _a.producto, _a.celda_precio, _a.celda_fecha, _a.hoja, _a.texto_fila


def _norm(t):
    return "".join(c for c in unicodedata.normalize("NFKD", str(t)) if not unicodedata.combining(c)).upper()

# ──────────────────────────────────────────────
# FUNCIONES
# ──────────────────────────────────────────────

def buscar_hoja(wb):
    """Busca la hoja indicada (--hoja) o la de Granos Básicos."""
    if HOJA:
        for nombre in wb.sheetnames:
            if _norm(HOJA) in _norm(nombre):
                return wb[nombre]
    for nombre in wb.sheetnames:
        if "granos" in nombre.lower() or "basicos" in nombre.lower() or "básicos" in nombre.lower():
            return wb[nombre]
    return wb.active


def leer_archivo(ruta):
    """Lee fecha (B3) y precio Maíz Blanco (D9) de un archivo Excel."""
    try:
        wb = load_workbook(ruta, data_only=True)
        hoja = buscar_hoja(wb)

        # Fecha en B3 — viene como datetime
        fecha = hoja[CELDA_FECHA].value
        if isinstance(fecha, datetime):
            dia, mes, anio = fecha.day, fecha.month, fecha.year
        else:
            return None, None, None, None

        # Precio en D9 (fila 9 = MAÍZ BLANCO)
        precio = hoja[CELDA_PRECIO].value
        if TEXTO_FILA:
            fila = re.sub(r"\D", "", CELDA_PRECIO) or "0"
            etiqueta = " ".join(str(hoja.cell(row=int(fila), column=c).value or "") for c in (1, 2, 3))
            if _norm(TEXTO_FILA) not in _norm(etiqueta):
                print(f"  ⚠️  la fila {fila} no contiene '{TEXTO_FILA}' (encontrado: '{etiqueta.strip()}')", end="  ")
        if precio is not None:
            try:
                precio = float(precio)
            except (ValueError, TypeError):
                precio = None

        return dia, mes, anio, precio

    except Exception as e:
        print(f"  ⚠️  Error en {os.path.basename(ruta)}: {e}")
        return None, None, None, None


# ──────────────────────────────────────────────
# PROCESO PRINCIPAL
# ──────────────────────────────────────────────

def main():
    print("=" * 60)
    print(f"  Consolidador de Precios - {PRODUCTO} (MAG El Salvador)")
    print("=" * 60)

    if not os.path.isdir(CARPETA_EXCEL):
        print(f"\n❌ No se encontró la carpeta: '{CARPETA_EXCEL}'")
        return

    archivos = sorted([
        f for f in os.listdir(CARPETA_EXCEL)
        if f.lower().endswith((".xlsx", ".xls", ".xlsm"))
    ])

    total = len(archivos)
    print(f"\n📂 Archivos encontrados: {total}\n")

    registros = []
    errores   = 0

    for i, nombre in enumerate(archivos, 1):
        ruta = os.path.join(CARPETA_EXCEL, nombre)
        print(f"[{i:>4}/{total}] {nombre}", end="  ")

        dia, mes, anio, precio = leer_archivo(ruta)

        if None in (dia, mes, anio, precio):
            print("⚠️  datos incompletos, omitido")
            errores += 1
            continue

        registros.append({
            "Producto": PRODUCTO,
            "Día":      dia,
            "Mes":      mes,
            "Año":      anio,
            "Precio":   precio,
        })
        print(f"✅  {dia:02d}/{mes:02d}/{anio}  →  {precio:.2f}")

    df = pd.DataFrame(registros, columns=["Producto", "Día", "Mes", "Año", "Precio"])
    df.sort_values(["Año", "Mes", "Día"], inplace=True, ignore_index=True)
    df.to_excel(ARCHIVO_SALIDA, index=False)

    print("\n" + "=" * 60)
    print(f"  ✅ Registros guardados  : {len(registros)}")
    print(f"  ⚠️  Archivos con error   : {errores}")
    print(f"  📄 Archivo de salida    : {os.path.abspath(ARCHIVO_SALIDA)}")
    print("=" * 60)


if __name__ == "__main__":
    main()