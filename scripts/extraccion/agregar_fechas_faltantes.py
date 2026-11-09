"""
Completa las fechas faltantes en un archivo Excel.
Agrega todos los días del calendario entre la fecha mínima y máxima,
dejando el Precio en blanco para los días faltantes.

Estructura esperada: Producto | Día | Mes | Año | Precio

Uso:
    python scripts/extraccion/agregar_fechas_faltantes.py ENTRADA.xlsx SALIDA.xlsx --producto "MAÍZ BLANCO"

Requisitos:
    pip install pandas openpyxl
"""

import argparse
import pandas as pd
from datetime import date, timedelta

# ─────────────────────────────────────
# CONFIGURACIÓN — ajusta estos valores
# ─────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("entrada", help="Excel original (Producto | Día | Mes | Año | Precio)")
ap.add_argument("salida", help="Excel resultado")
ap.add_argument("--producto", required=True, help="Valor de Producto para las filas nuevas")
_a = ap.parse_args()
ARCHIVO_ENTRADA, ARCHIVO_SALIDA, PRODUCTO = _a.entrada, _a.salida, _a.producto

# ─────────────────────────────────────
# LEER ARCHIVO
# ─────────────────────────────────────
print(f"📂 Leyendo: {ARCHIVO_ENTRADA}")
df = pd.read_excel(ARCHIVO_ENTRADA)

print(f"   Registros originales : {len(df)}")
print(f"   Columnas detectadas  : {list(df.columns)}")

# Normalizar nombres de columnas (quitar espacios)
df.columns = df.columns.str.strip()

# Asegurar tipos correctos
df["Día"]  = df["Día"].astype(int)
df["Mes"]  = df["Mes"].astype(int)
df["Año"]  = df["Año"].astype(int)

# Crear columna de fecha para facilitar comparación
df["_fecha"] = pd.to_datetime(df[["Año","Mes","Día"]].rename(
    columns={"Año":"year","Mes":"month","Día":"day"}
))

# ─────────────────────────────────────
# GENERAR RANGO COMPLETO DE FECHAS
# ─────────────────────────────────────
fecha_min = df["_fecha"].min()
fecha_max = df["_fecha"].max()

print(f"   Rango detectado      : {fecha_min.date()} → {fecha_max.date()}")

rango_completo = pd.date_range(start=fecha_min, end=fecha_max, freq="D")

# Fechas que ya existen
fechas_existentes = set(df["_fecha"])

# Filas nuevas para fechas faltantes
nuevas = []
for fecha in rango_completo:
    if fecha not in fechas_existentes:
        nuevas.append({
            "Producto": PRODUCTO,
            "Día":      fecha.day,
            "Mes":      fecha.month,
            "Año":      fecha.year,
            "Precio":   None,   # en blanco
            "_fecha":   fecha,
        })

print(f"   Fechas faltantes     : {len(nuevas)}")

# ─────────────────────────────────────
# COMBINAR Y ORDENAR
# ─────────────────────────────────────
if nuevas:
    df_nuevas = pd.DataFrame(nuevas)
    df_final  = pd.concat([df, df_nuevas], ignore_index=True)
else:
    df_final = df.copy()

df_final.sort_values("_fecha", inplace=True, ignore_index=True)

# Eliminar columna auxiliar
df_final.drop(columns=["_fecha"], inplace=True)

# ─────────────────────────────────────
# GUARDAR
# ─────────────────────────────────────
df_final.to_excel(ARCHIVO_SALIDA, index=False)

print(f"\n✅ Registros en resultado : {len(df_final)}")
print(f"📄 Archivo guardado en    : {ARCHIVO_SALIDA}")