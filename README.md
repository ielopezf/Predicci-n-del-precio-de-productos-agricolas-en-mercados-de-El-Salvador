# Predicción del precio de productos agrícolas en mercados de El Salvador

**Aplicación publicada:** https://prediccionpreciosagricolassv.streamlit.app/


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
10. [Etapa final: reproducir los resultados de modelado](#10-etapa-final-reproducir-los-resultados-de-modelado)

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
| `lluvia_mensual_sv.xlsx` | `data/external/` | Lluvia mensual (opcional: solo la usan la auditoría y la prueba a10) |

> Si `data/interim/precios_consolidados.xlsx` no existe, descarga los informes diarios del MAG con
> `scripts/extraccion/descargar_mag.py` y consolídalos con el notebook `00_consolidacion_datos.ipynb`
> (ver sección 12).

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
├── scripts/                    # análisis de la etapa final (a1…a11) y run_all.py — ver sección 10
├── tests/                      # pruebas automáticas (pytest)
├── models/                     # modelos entrenados (.pkl / .keras) — generados
├── api/                        # API Flask de despliegue (sirve los modelos) — ver sección 9
├── streamlit_app/              # prototipo Streamlit (consume la API) — ver sección 9
├── reports/
│   ├── figures/                # figuras del EDA y resultados — generadas
│   ├── etapa1/                 # documento de resultados de la Etapa 1
│   ├── etapa2/                 # tabla comparativa, hiperparámetros, reporte de equidad — generados
│   └── etapa3/                 # resultados de la etapa final (CSV) — generados por scripts/
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
| `baselines_ingenuos.py` | Persistencia, media móvil, estacional, deriva, ETS amortiguado y ARIMA(p,1,0) (NumPy/SciPy) — etapa final |
| `auditoria_datos.py` | Auditoría de calidad: nulos, repeticiones de precio, recortes y cobertura de fuentes — etapa final |
| `pipeline_causal.py` | Dataset semanal causal: `precio_imputado`, `n_dias`, `semanas_sin_cambio`, sin interpolar hacia el futuro — etapa final |

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

## 10. Etapa final: reproducir los resultados de modelado

La etapa final revisa el tratamiento de datos y compara los modelos con una evaluación
equivalente (misma división temporal, mismas variables y mismo presupuesto de afinación).
Todos los resultados se generan como CSV en `reports/etapa3/` con un solo comando.

```bash
# Con el entorno virtual activo y desde la raíz del repositorio
python scripts/run_all.py                              # usa data/interim/precios_consolidados.xlsx
python scripts/run_all.py --precios RUTA/al/excel.xlsx # si el archivo está en otra ubicación
python scripts/run_all.py --con-lstm                   # incluye el LSTM (requiere tensorflow; lento)
python scripts/run_all.py --solo a4,a6                 # solo algunos pasos
```

`run_all.py` corre primero las pruebas (`pytest`) y después cada paso en orden; se
detiene al primer error e indica cómo reanudar con `--solo`. Requiere
`data/processed/dataset_modelado.csv` y el Excel de precios. Si falta
`data/external/lluvia_mensual_sv.xlsx`, omite el paso a10.

| Clave | Script | Qué hace | Salidas principales en `reports/etapa3/` |
|---|---|---|---|
| a1 | `a1_baselines.py` | Referencias ingenuas, ETS y ARIMA sobre las 157 filas de prueba | `metricas_baselines.csv` |
| a2 | `a2_auditoria.py` | Auditoría de datos y cobertura de fuentes | `auditoria_precios.csv`, `cobertura_fuentes.csv`, `semanas_sin_dato.csv` |
| a9 | `a9_sensibilidad_nulos.py` | Sensibilidad al tratamiento de nulos y valores atípicos | `sensibilidad_nulos.csv` |
| a4 | `a4_horizontes.py` | Horizontes de 1, 2, 4 y 8 semanas con origen móvil | `horizontes_resumen.csv` |
| a6 | `a6_afinacion.py` | Afinación equivalente de todos los modelos | `tabla_por_producto.csv`, `afinacion_cv.csv` |
| a7 | `a7_ensamble.py` | Ensamble contraído hacia la persistencia (necesita a6) | `ensamble_resumen.csv` |
| a8 | `a8_inferencia.py` | Diebold-Mariano e intervalos de predicción (necesita a6 y a7) | `diebold_mariano.csv`, `intervalos_persistencia.csv` |
| a10 | `a10_prueba_lluvia_mensual.py` | ¿Aporta la lluvia mensual? (necesita a6) | `prueba_lluvia_mensual.csv` |
| a11 | `a11_lstm.py` | LSTM con el mismo protocolo (opcional; usa funciones de a6 pero no sus resultados) | `lstm_resumen.csv` |

**Resultados principales** (prueba: 26-nov-2023 a 30-jun-2024; periodo de datos: mayo de 2021 a junio de 2024):

- La persistencia (el precio de la semana pasada) es la mejor referencia: MAPE global de
  4.49 % a una semana y 16.24 % a ocho.
- Con afinación equivalente, regresión lineal, Ridge, Random Forest y Gradient Boosting no la
  superan de forma consistente; la prueba de Diebold-Mariano los da como significativamente peores.
- Cambiar el tratamiento de nulos y valores atípicos modifica el MAPE global en menos de 0.6 puntos.
- El error depende del producto: el arroz es casi constante; el tomate tiene un cambio semanal medio de 15 %.
- Ni la lluvia diaria ni la mensual mejoran los pronósticos.

**Nota sobre el LSTM.** `a11_lstm.py` necesita `tensorflow` o `keras` (están en
`requirements.txt`). Con `--stub` solo comprueba el flujo de datos con una regresión Ridge,
sin red neuronal; esos resultados **no** son del LSTM.

**Pruebas automáticas:** `python -m pytest tests/test_baselines.py`.

**Limitaciones:** los precios llegan a junio de 2024 y no se encontraron fuentes más recientes;
la prueba tiene 28 semanas por producto; los precios de la naranja se repiten en el 87 % de los
días. Los notebooks 03 y 04 conservan los resultados de la Etapa 2, que no son comparables con
las tablas de esta sección.

---

*Proyecto académico. Datos de fuentes públicas oficiales de El Salvador (MAG, BCR,
DIGESTYC, DGEHM) y FAOSTAT.*


## 11. Aplicación: pronóstico a 1–4 semanas

- **Módulo**: `src/prediccion_precios/pronostico.py` (funciones puras, probadas en `tests/test_pronostico.py`).
- **Método**: persistencia (último precio) + intervalos empíricos 80/90 % + semáforo de confiabilidad por producto y horizonte + avisos (tomate, meses de cosecha, dato viejo). Es lo que respalda el informe: ningún modelo supera a la persistencia de forma robusta.
- **Exportar**: `python scripts/a12_pronostico.py [--mag reporte.csv]` → `reports/pronostico.csv`.
- **Aplicación publicada**: https://prediccionpreciosagricolassv.streamlit.app/ (QR: `python scripts/generar_qr.py`, requiere `pip install "qrcode[pil]"`).
- **Streamlit (local)**: `streamlit run streamlit_app/app.py` → vista "Pronóstico (1–4 semanas)"; permite cargar precio manual o un CSV/XLSX con columnas `Fecha, Producto, Precio`.
- **API**: `GET /api/pronostico[?producto=TOMATE&precio=30&fecha_base=2025-01-05]`.
- **Limitación**: los datos terminan en junio de 2024; sin precio vigente la app avisa que el pronóstico corresponde a esa fecha. Los intervalos cubrieron 63 % (nominal 80 %) y 73 % (nominal 90 %) en la prueba.
- **CI**: `.github/workflows/tests.yml` corre pytest en cada push.

## 12. Extracción de datos (opcional)

Scripts para volver a descargar las fuentes. No se necesitan para la app ni para reproducir los modelos si ya tienes `data/`.
Instalar: `pip install -r requirements-extraccion.txt` (Selenium requiere Chrome).

| Script | Fuente | Salida |
|---|---|---|
| `scripts/extraccion/descargar_mag.py` | Informe diario de precios del MAG (Excel) | carpeta `excels_mag/` |
| `scripts/extraccion/obtener_precipitaciones_snet.py` | SNET/MARN, lluvia diaria por estación | `precipitaciones_el_salvador_2020_2026.xlsx/.csv` |
| `scripts/extraccion/obtener_precios_combustible.py` | SINAPP-DGEHM, combustibles | `precios_combustibles_sv.xlsx` |
| `scripts/extraccion/consolidar_precio_mag.py` | Excel del MAG en `excels_mag/` | tabla de un producto (Producto, Día, Mes, Año, Precio); lee celdas fijas, ver su docstring |
| `scripts/extraccion/agregar_fechas_faltantes.py` | — | completa días sin precio de una serie (`ENTRADA SALIDA --producto`) |

Nota: en el extractor de combustibles, un valor vacío en la fuente se guarda como `0.0` (no como nulo). Eso explica los ceros de diésel que reporta la auditoría; los modelos los reciben tal cual vienen en `data/external/`.
