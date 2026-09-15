# Predicción del precio de productos agrícolas en mercados de El Salvador

Modelo predictivo del comportamiento futuro de los precios de productos agrícolas
seleccionados (maíz, frijol, arroz, tomate y naranja) a partir de datos históricos
oficiales, siguiendo la metodología **CRISP-DM**. Meta del objetivo general:
**MAPE < 15%**.

Curso de Especialización en **Machine Learning** — Universidad de El Salvador,
Facultad de Ingeniería y Arquitectura, Escuela de Ingeniería de Sistemas
Informáticos. Ciclo I 2026. **Grupo 01.** Docente: Ing. Bladimir Díaz Campos.

**Integrantes:** Abrego Guillén, Maynor Josué (AG17020) · López Fuentes, Irma Elena
(LF18003) · Serrano Fuentes, Elena Sarai (SF18004).

---

## Tabla de contenido
1. [Requisitos previos](#1-requisitos-previos)
2. [Instalación](#2-instalación)
3. [Datos necesarios](#3-datos-necesarios)
4. [Cómo ejecutar el proyecto](#4-cómo-ejecutar-el-proyecto)
5. [Estructura del proyecto](#5-estructura-del-proyecto)
6. [El paquete `prediccion_precios`](#6-el-paquete-prediccion_precios)
7. [Flujo CRISP-DM y cobertura de la rúbrica](#7-flujo-crisp-dm-y-cobertura-de-la-rúbrica)
8. [Solución de problemas](#8-solución-de-problemas)
9. [Despliegue: API + Streamlit](#9-despliegue-api--streamlit)

---

## 1. Requisitos previos

- **Python 3.10 o superior** ([descargar](https://www.python.org/downloads/)).
- **Git** para clonar el repositorio.
- Sistema operativo: Windows, macOS o Linux.
- No se necesita GPU para la Etapa 1 (los baselines corren en CPU).

Verifica tu versión de Python:

```bash
python --version      # debe mostrar 3.10 o superior
```

## 2. Instalación

**Paso 1 — Obtener el proyecto.** Clona el repositorio (o descarga y descomprime el ZIP):

```bash
git clone <URL-del-repositorio>
cd prediccion_precios_agricolas
```

**Paso 2 — Crear un entorno virtual.** Aísla las dependencias del proyecto:

```bash
python -m venv .venv
```

Actívalo:

```bash
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# Windows (CMD)
.venv\Scripts\activate.bat
# macOS / Linux
source .venv/bin/activate
```

**Paso 3 — Instalar las dependencias:**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Paso 4 — Registrar el kernel de Jupyter** (para que los notebooks usen este entorno):

```bash
python -m ipykernel install --user --name prediccion-precios
```

## 3. Datos necesarios

El proyecto usa datos públicos oficiales de El Salvador. Los archivos ya deben estar
ubicados así (algunos vienen incluidos en el repositorio):

| Archivo | Ubicación | Fuente |
|---|---|---|
| `precios_consolidados.xlsx` | `data/interim/` | MAG — precios de mercados mayoristas (consolidado) |
| `precios_combustibles_sv.xlsx` | `data/external/` | Dir. Gral. de Energía, Hidrocarburos y Minas |
| `precipitaciones_el_salvador_2021-2024.xlsx` | `data/external/` | Registros de precipitación diaria |
| `ipc_transporte.csv` | `data/external/` | BCR — IPC partición 1.7 Transporte |
| `Produccion.csv` | `data/external/` | FAOSTAT — producción agrícola anual |

> Si `data/interim/precios_consolidados.xlsx` no existe, genéralo primero a partir de
> los informes diarios del MAG con el script `Extraer datos/extraer_precios.py`
> (ver notebook `00_consolidacion_datos.ipynb`).

## 4. Cómo ejecutar el proyecto

Inicia Jupyter:

```bash
jupyter notebook       # o: jupyter lab
```

Ejecuta los notebooks **en orden**, seleccionando el kernel `prediccion-precios`:

| Orden | Notebook | Qué hace | Salidas |
|---|---|---|---|
| 1 | `00_consolidacion_datos.ipynb` | Verifica/documenta el consolidado del MAG | — |
| 2 | `01_eda.ipynb` | Análisis exploratorio (EDA) | figuras en `reports/figures/` |
| 3 | `02_preprocesamiento_features.ipynb` | Limpieza, integración y features | `data/processed/dataset_modelado.csv` |
| 4 | `03_modelos_baseline.ipynb` | Entrena 3 baselines, métricas, SHAP | tabla de métricas, figuras, modelos en `models/` |
| 5 | `04_modelos_avanzados.ipynb` | HPO (Optuna), Red Neuronal, LSTM, ensemble, CV rigurosa + prueba final, equidad/ética | modelos avanzados en `models/`, `reports/etapa2/` |

Cada notebook importa la lógica desde `src/prediccion_precios/`, por lo que el código
queda limpio, versionado y reproducible.

Con los modelos ya entrenados (notebooks 03 y 04), la app de despliegue (API + Streamlit)
se levanta como se explica en la [sección 9](#9-despliegue-api--streamlit).

**Ejecución rápida desde la terminal** (sin abrir Jupyter), para regenerar el dataset:

```bash
python -c "import sys; sys.path.insert(0,'src'); \
from prediccion_precios import data_loading as dl, preprocessing as pp, features as ft; \
d = dl.filtrar_productos_objetivo(dl.cargar_precios_consolidados()); \
d = pp.tratar_faltantes(pp.resamplear_frecuencia(pp.tratar_outliers(pp.limpiar_precios(d)))); \
d = pp.integrar_fuentes(d, dl.cargar_combustible(), dl.cargar_precipitacion(), dl.cargar_ipc(), dl.cargar_produccion()); \
d = ft.marcar_temporada_cosecha(ft.agregar_medias_moviles(ft.agregar_lags(ft.agregar_variables_calendario(d)))); \
print('dataset:', pp.guardar_processed(d))"
```

## 5. Estructura del proyecto

```
prediccion_precios_agricolas/
├── data/
│   ├── raw/          # Informes diarios del MAG (.xlsx) sin procesar
│   ├── interim/      # precios_consolidados.xlsx (11,648 filas, 16 productos, 2021-2024)
│   ├── processed/    # dataset_modelado.csv (generado por el notebook 02)
│   └── external/     # combustible, precipitación, IPC transporte, producción FAOSTAT
├── notebooks/
│   ├── 00_consolidacion_datos.ipynb
│   ├── 01_eda.ipynb
│   ├── 02_preprocesamiento_features.ipynb
│   ├── 03_modelos_baseline.ipynb
│   └── 04_modelos_avanzados.ipynb
├── src/prediccion_precios/     # paquete Python con toda la lógica
├── models/                     # modelos entrenados (.pkl / .keras) — generados
├── api/                        # API Flask de despliegue (sirve los modelos) — ver sección 9
├── streamlit_app/              # prototipo Streamlit (consume la API) — ver sección 9
├── reports/
│   ├── figures/                # figuras del EDA y resultados — generadas
│   ├── etapa1/                 # documento de resultados de la Etapa 1
│   └── etapa2/                 # tabla comparativa, hiperparámetros, reporte de equidad — generados
├── references/                 # papers y documentación consultada
├── requirements.txt
├── RECURSOS.md                 # recursos del proyecto completo
├── .gitignore
└── README.md
```

## 6. El paquete `prediccion_precios`

| Módulo | Responsabilidad |
|---|---|
| `config.py` | Rutas, nombres de columnas, productos objetivo, calendario de cosecha, semilla, meta MAPE |
| `data_loading.py` | Carga y describe la fuente principal y las 4 fuentes externas |
| `eda.py` | Estadística descriptiva, series, estacionalidad, distribución, outliers, correlación |
| `preprocessing.py` | Limpieza, outliers, resampleo semanal, relleno de huecos, integración de fuentes |
| `features.py` | Calendario cíclico, lags, medias móviles, temporada de cosecha, matriz de modelado |
| `models_baseline.py` | Regresión Lineal, Random Forest y XGBoost (Etapa 1) |
| `models_advanced.py` | XGBoost optimizado (Optuna/GridSearchCV), Red Neuronal y LSTM (Keras), ensemble (voting/stacking) — Etapa 2 |
| `evaluation.py` | Métricas MAE/RMSE/MAPE/R², split temporal (2 y 3 tramos), validación cruzada temporal, tabla comparativa |
| `interpretability.py` | Importancia de variables, SHAP y LIME |
| `fairness.py` | Evaluación ética y de sesgos: métricas por producto/temporada, brecha de equidad, limitaciones — Etapa 2 |

Las dependencias pesadas (scikit-learn, xgboost, shap, lime, optuna, tensorflow) se
importan de forma diferida, por lo que el paquete se importa sin error aunque falten;
solo se exigen al llamar a la función correspondiente.

## 7. Flujo CRISP-DM y cobertura de la rúbrica

| Fase CRISP-DM | Dónde vive |
|---|---|
| Comprensión del negocio | Perfil de trabajo + `config.py` |
| Comprensión de los datos | `notebooks/00`, `notebooks/01`, `data_loading.py`, `eda.py` |
| Preparación de los datos | `notebooks/02`, `preprocessing.py`, `features.py` |
| Modelado | `notebooks/03` + `models_baseline.py` (Etapa 1); `notebooks/04` + `models_advanced.py` (Etapa 2: HPO, LSTM, red neuronal, ensemble) |
| Evaluación | `evaluation.py`, `interpretability.py`, `fairness.py` (equidad/ética, Etapa 2) |
| Despliegue | `api/` (Flask) + `streamlit_app/` (Streamlit) — ver [sección 9](#9-despliegue-api--streamlit) |

**Rúbrica Etapa 1 (100 pts, 25% de la nota):**

| Criterio | Pts | Dónde se resuelve |
|---|---:|---|
| Calidad y cantidad de datos | 20 | `data_loading.py`, `notebooks/00` |
| EDA completo y visualizaciones | 20 | `eda.py`, `notebooks/01`, `reports/figures/` |
| Preprocesamiento y feature engineering | 20 | `preprocessing.py`, `features.py`, `notebooks/02` |
| Modelos baseline y métricas (≥3 + tabla) | 20 | `models_baseline.py`, `evaluation.py`, `notebooks/03` |
| Interpretabilidad inicial (SHAP/LIME) | 10 | `interpretability.py`, `notebooks/03` |
| Código reproducible (GitHub) | 10 | estructura, `requirements.txt`, `.gitignore` |

## 8. Solución de problemas

- **`ModuleNotFoundError: No module named 'prediccion_precios'`** — Ejecuta los
  notebooks desde la carpeta `notebooks/` (agregan `../src` al path automáticamente) o
  añade `src/` a tu `PYTHONPATH`.
- **`ModuleNotFoundError: No module named 'xgboost' / 'sklearn' / 'shap'`** — Faltan las
  dependencias; ejecuta `pip install -r requirements.txt` dentro del entorno virtual.
- **El kernel de Jupyter no encuentra las librerías** — Asegúrate de seleccionar el
  kernel `prediccion-precios` (paso 4 de instalación) y de haber activado el entorno.
- **`FileNotFoundError` al cargar datos** — Verifica que los archivos de la sección 3
  estén en las rutas indicadas.
- **Los precios muestran muchos faltantes** — Está previsto: se seleccionaron, por
  cultivo, las variantes con mayor cobertura (~0.3%). Ver `config.PRODUCTOS_OBJETIVO`.

## 9. Despliegue: API + Streamlit

Dos carpetas independientes de `src/prediccion_precios/` (que solo importan del paquete,
no lo modifican):

- **`api/`** — API Flask que carga los modelos entrenados en `models/` y expone
  endpoints de solo lectura en modo **backtest** (precio real vs. predicho sobre el
  tramo de test histórico que ningún modelo vio al entrenarse):
  `GET /api/productos`, `/api/modelos`, `/api/prediccion?modelo=...&productos=slug1,slug2`,
  `/api/metricas?modelo=...`, `/api/importancia?modelo=...`, `/api/equidad?modelo=...`.
- **`streamlit_app/`** — prototipo Streamlit que consume la API: permite elegir uno o
  varios productos y un modelo, y muestra las series real vs. predicho, la tabla de
  métricas, la importancia de variables y la evaluación de equidad/ética.

No ofrecen pronóstico hacia el futuro: no existen valores reales futuros de
combustible/IPC/clima con los que alimentar al modelo, así que inventarlos sería
engañoso (ver `fairness.LIMITACIONES_ETICAS`).

**Ejecutar (requiere los modelos ya entrenados — notebooks 03 y 04):**

```bash
# Terminal 1 — API
cd api
python app.py                       # sirve en http://localhost:5000

# Terminal 2 — Streamlit
cd streamlit_app
streamlit run app.py                # abre http://localhost:8501
```

Por defecto Streamlit apunta a `http://localhost:5000`; para usar otra URL de API,
define la variable de entorno `API_URL` antes de ejecutar `streamlit run`.

---

*Proyecto académico. Datos de fuentes públicas oficiales de El Salvador (MAG, BCR,
DIGESTYC, DGEHM) y FAOSTAT.*
