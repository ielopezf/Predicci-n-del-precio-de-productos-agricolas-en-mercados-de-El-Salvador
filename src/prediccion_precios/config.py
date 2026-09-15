"""
config.py — Rutas, constantes y parámetros globales del proyecto.

Centraliza todo lo que "cambia según el entorno" (rutas) o que se repite en
varios módulos (nombres de columnas, productos objetivo, semilla aleatoria).
Así ningún notebook contiene rutas absolutas y el proyecto es reproducible en
cualquier máquina del equipo.
"""

from pathlib import Path

# --------------------------------------------------------------------------- #
# Rutas base (se resuelven de forma relativa a la raíz del repositorio)
# --------------------------------------------------------------------------- #
ROOT_DIR = Path(__file__).resolve().parents[2]      # .../prediccion_precios_agricolas
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"                            # datos crudos (informes MAG, xlsx)
INTERIM_DIR = DATA_DIR / "interim"                    # datos intermedios (consolidado)
PROCESSED_DIR = DATA_DIR / "processed"               # dataset final listo para modelar
EXTERNAL_DIR = DATA_DIR / "external"                 # combustibles, producción, IPC, etc.
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"                # gráficos del EDA (obligatorios)
ETAPA2_DIR = REPORTS_DIR / "etapa2"                  # tabla comparativa, hiperparámetros, equidad
MODELS_DIR = ROOT_DIR / "models"                     # modelos serializados (.pkl / .keras)

# --------------------------------------------------------------------------- #
# Fuente de datos principal (ya consolidada por el equipo)
# --------------------------------------------------------------------------- #
# El archivo precios_consolidados.xlsx contiene 11,648 registros, 16 productos,
# años 2021-2024, columnas: Archivo, Hoja, Producto, Precio, Fecha, Dia, Mes, Anio.
PRECIOS_CONSOLIDADOS = INTERIM_DIR / "precios_consolidados.xlsx"
HOJA_PRECIOS = "Precios"

# --------------------------------------------------------------------------- #
# Nombres de columnas (evita "strings mágicos" repartidos por el código)
# --------------------------------------------------------------------------- #
COL_PRODUCTO = "Producto"
COL_PRECIO = "Precio"
COL_FECHA = "Fecha"
COL_CATEGORIA = "Hoja"        # GranosBasicos | Hortalizas | Frutas
COL_DIA = "Dia"
COL_MES = "Mes"
COL_ANIO = "Anio"
TARGET = COL_PRECIO           # variable objetivo (regresión)

# --------------------------------------------------------------------------- #
# Productos objetivo del estudio (según el perfil de trabajo)
# --------------------------------------------------------------------------- #
# Se eligió, por grupo de cultivo, la variante con MAYOR cobertura de datos
# (~0.3% de faltantes). Se descartaron variantes con datos casi vacíos, p. ej.
# ARROZ ... NACIONAL (59.8% faltantes) y TOMATE DE ENSALADA GRANDE (94.4%).
# Los riesgos del perfil advierten no intentar predecir demasiados productos a la vez.
PRODUCTOS_OBJETIVO = [
    "MAÍZ BLANCO",                       # 0.3% faltantes
    "FRIJOL  ROJO DE SEDA NACIONAL",     # 0.3% faltantes
    "ARROZ ORO PRIMERA CLASE IMPORTADO", # 0.3% faltantes (la variante nacional tiene 59.8%)
    "TOMATE DE PASTA GRANDE",            # 0.3% faltantes (ensalada grande tiene 94.4%)
    "NARANJA VALENCIA MEDIANA",          # 0.4% faltantes
]

# --------------------------------------------------------------------------- #
# Fuentes externas a integrar (Comprensión/Preparación de datos - CRISP-DM)
# --------------------------------------------------------------------------- #
# Archivos colocados por el equipo en data/external/.
FUENTE_COMBUSTIBLE = EXTERNAL_DIR / "precios_combustibles_sv.xlsx"      # diario, por región y tipo
FUENTE_PRECIPITACION = EXTERNAL_DIR / "precipitaciones_el_salvador_2021-2024.xlsx"  # diario
FUENTE_IPC = EXTERNAL_DIR / "ipc_transporte.csv"   # mensual (IPC 1.7 Transporte, Dic 2009=100)
FUENTE_PRODUCCION = EXTERNAL_DIR / "Produccion.csv"                      # anual (FAOSTAT)

# Mapeo producto agrícola -> "Item" de producción FAOSTAT (para unir por cultivo).
PRODUCTO_A_ITEM_PRODUCCION = {
    "MAÍZ": "Maize (corn)",
    "FRIJOL": "Beans, dry",
    "ARROZ": "Rice",
    "TOMATE": "Tomatoes",
    "NARANJA": "Oranges",
}

# --------------------------------------------------------------------------- #
# Preparación de datos
# --------------------------------------------------------------------------- #
FRECUENCIA = "W"                      # frecuencia común (semanal) para alinear fuentes
DATASET_MODELADO = PROCESSED_DIR / "dataset_modelado.csv"

# Meses de cosecha por cultivo en El Salvador (aporte local; ver perfil, calendario
# agrícola: primera, postrera y apante). Usado para la bandera `es_cosecha`.
MESES_COSECHA = {
    "MAÍZ": [8, 9, 11, 12],       # primera (ago-sep) y postrera (nov-dic)
    "FRIJOL": [2, 3, 11, 12],     # apante (feb-mar) y postrera (nov-dic)
    "ARROZ": [11, 12, 1],         # cosecha nov-ene
    "TOMATE": [11, 12, 1, 2, 3],  # producción bajo riego, temporada seca
    "NARANJA": [11, 12, 1, 2],    # temporada de naranja nov-feb
}

# --------------------------------------------------------------------------- #
# Parámetros de modelado y evaluación
# --------------------------------------------------------------------------- #
RANDOM_STATE = 42
TEST_SIZE = 0.2               # split temporal, NO aleatorio (serie de tiempo)
CV_SPLITS = 5                 # TimeSeriesSplit para validación cruzada
META_MAPE_OBJETIVO = 0.15     # meta del objetivo general: MAPE < 15%
