# Recursos necesarios — Proyecto completo

Detalle de recursos para llevar el proyecto de principio a fin (Etapas 1 y 2 + defensa).
Consolida y amplía la sección "Recursos necesarios" del perfil de trabajo.

## 1. Software y librerías

| Recurso | Uso | Etapa |
|---|---|---|
| Python 3.10+ | Lenguaje base del proyecto | Todas |
| Jupyter Notebook / Google Colab | Desarrollo interactivo de EDA y modelos | Todas |
| pandas, numpy, openpyxl, pyarrow | Manipulación y consolidación de datos | 1 |
| matplotlib, seaborn | Visualizaciones del EDA | 1 |
| scikit-learn | Baselines (Reg. Lineal, Random Forest), métricas, CV temporal | 1 |
| XGBoost | Modelo baseline fuerte y candidato final | 1–2 |
| SHAP / LIME | Interpretabilidad de modelos | 1–2 |
| TensorFlow / Keras | LSTM y Redes Neuronales | 2 |
| Optuna / GridSearchCV | Optimización de hiperparámetros | 2 |
| Streamlit o Flask | Prototipo funcional desplegable | 2 |
| Git + GitHub | Control de versiones y trabajo colaborativo | Todas |
| Excel | Revisión rápida de datos crudos del MAG | 1 |

## 2. Datos

| Fuente | Institución | Contenido | Formato |
|---|---|---|---|
| Precios de mercados mayoristas | MAG | Precios diarios/semanales de productos agrícolas (ya consolidados) | .xlsx → .csv/.parquet |
| Precios de combustible | Dir. Gral. de Energía, Hidrocarburos y Minas | Costo de transporte (variable explicativa) | .csv/.xlsx |
| Producción agrícola anual | MAG (Anuario de Estadísticas Agropecuarias) | Superficie, producción, rendimiento | .xlsx/PDF |
| IPC / canasta básica | BCR / DIGESTYC | Indicadores macroeconómicos | Web / .xlsx |

**Nota de licencia:** todas son fuentes públicas oficiales de El Salvador; documentar la
condición de uso concreta de cada una en el reporte de Etapa 1.

## 3. Hardware

- Equipos portátiles personales del equipo (desarrollo de Etapa 1: EDA y baselines corren
  sin problema en CPU).
- Conectividad a internet para descargar datos y usar servicios en la nube.
- **Google Colab (GPU gratuita)** recomendado para el entrenamiento de LSTM/Redes Neuronales
  en la Etapa 2, evitando la necesidad de GPU local.

## 4. Recursos humanos

- Tres especialistas en formación (roles de ingeniería de datos y modelado).
- Asesoría del docente para validación metodológica e interpretación de métricas de error.
- **Restricción a considerar (riesgo del perfil):** el equipo tiene jornadas laborales de
  tiempo completo → mantener el alcance acotado (pocos productos objetivo, cronograma realista
  de 7 meses).

## 5. Servicios y plataformas

- Repositorio GitHub (evidencia obligatoria de la rúbrica de Etapa 1).
- Google Drive / Colab para datos y cómputo compartido.
- Herramienta de gestión de tareas (Trello, Notion o el diagrama de Gantt del perfil) para el
  seguimiento del cronograma.
