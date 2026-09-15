"""models_advanced.py — Modelos avanzados (CRISP-DM: Modelado - Etapa 2).

Cubre los 4 puntos obligatorios de la Etapa 2 sobre los baselines de
`models_baseline.py`:

    * XGBoost optimizado -> `optimizar_xgboost` (Optuna / GridSearchCV).
    * Redes Neuronales   -> `RedNeuronalRegresora` (MLP denso, Keras).
    * LSTM               -> `LSTMRegresor` + `preparar_secuencias` (Keras).
    * Ensemble           -> `construir_ensemble` (voting / stacking).

Las dependencias pesadas (sklearn, xgboost, optuna, tensorflow) se importan de
forma DIFERIDA dentro de cada función, para que el módulo se importe sin error
aunque falten (se exigen solo al llamar a la función correspondiente).

Todas las búsquedas de hiperparámetros usan `TimeSeriesSplit` (ventana
expansiva) SOLO sobre el tramo de train — el tramo de test se reserva para la
evaluación final única ("prueba final rigurosa", ver `evaluation.split_train_val_test`).
"""

from __future__ import annotations

from . import config


# --------------------------------------------------------------------------- #
# 1. Optimización de hiperparámetros
# --------------------------------------------------------------------------- #
def optimizar_xgboost(X_train, y_train, metodo: str = "optuna", n_trials: int = 30,
                       cv_splits: int | None = None):
    """Búsqueda de hiperparámetros para XGBoost sobre el tramo de train.

    metodo="optuna": búsqueda bayesiana (TPE) minimizando el MAPE medio en CV
    temporal. metodo="gridsearch": grilla exhaustiva con `GridSearchCV`
    (alternativa si Optuna no está disponible).

    Returns: (modelo_final_ajustado, info) — info trae los mejores
    hiperparámetros y el score de validación cruzada, para dejar evidencia.
    """
    import numpy as np
    from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
    from . import models_baseline as mb, evaluation as ev

    cv_splits = cv_splits or config.CV_SPLITS
    tscv = TimeSeriesSplit(n_splits=cv_splits)
    Xtr = X_train.reset_index(drop=True) if hasattr(X_train, "reset_index") else X_train
    ytr = y_train.reset_index(drop=True) if hasattr(y_train, "reset_index") else y_train

    if metodo == "optuna":
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def _objetivo(trial):
            params = dict(
                n_estimators=trial.suggest_int("n_estimators", 100, 600, step=50),
                max_depth=trial.suggest_int("max_depth", 2, 8),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                subsample=trial.suggest_float("subsample", 0.6, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
                min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
                reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            )
            mapes = []
            for idx_tr, idx_val in tscv.split(Xtr):
                Xf = Xtr.iloc[idx_tr] if hasattr(Xtr, "iloc") else Xtr[idx_tr]
                Xv = Xtr.iloc[idx_val] if hasattr(Xtr, "iloc") else Xtr[idx_val]
                yf = ytr.iloc[idx_tr] if hasattr(ytr, "iloc") else ytr[idx_tr]
                yv = ytr.iloc[idx_val] if hasattr(ytr, "iloc") else ytr[idx_val]
                modelo = mb.nuevo_xgboost(**params).fit(Xf, yf)
                mapes.append(ev.calcular_metricas(yv, modelo.predict(Xv))["MAPE_%"])
            return float(np.mean(mapes))

        study = optuna.create_study(direction="minimize",
                                     sampler=optuna.samplers.TPESampler(seed=config.RANDOM_STATE))
        study.optimize(_objetivo, n_trials=n_trials, show_progress_bar=False)
        mejores = study.best_params
        modelo_final = mb.nuevo_xgboost(**mejores).fit(X_train, y_train)
        info = {"metodo": "optuna", "n_trials": n_trials,
                "mejores_parametros": mejores, "mejor_mape_cv": round(study.best_value, 4)}
        return modelo_final, info

    if metodo == "gridsearch":
        grid = {
            "n_estimators": [200, 400, 600],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.03, 0.05, 0.1],
            "subsample": [0.8, 1.0],
        }
        gs = GridSearchCV(mb.nuevo_xgboost(), grid, cv=tscv,
                           scoring="neg_mean_absolute_percentage_error", n_jobs=-1)
        gs.fit(X_train, y_train)
        info = {"metodo": "gridsearch", "mejores_parametros": gs.best_params_,
                "mejor_score_cv": round(float(gs.best_score_), 4)}
        return gs.best_estimator_, info

    raise ValueError('metodo debe ser "optuna" o "gridsearch".')


# --------------------------------------------------------------------------- #
# 2. Red neuronal (MLP denso)
# --------------------------------------------------------------------------- #
def construir_red_neuronal(input_dim: int):
    """Arquitectura de una red neuronal densa (MLP) para regresión de precio."""
    from tensorflow import keras
    modelo = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(1),
    ])
    modelo.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return modelo


class RedNeuronalRegresora:
    """MLP (Keras) con escalado de X/y integrado e interfaz fit/predict tipo sklearn.

    El escalado es necesario porque las features mezclan escalas muy distintas
    (p. ej. `produccion_t` en miles vs. `mes_sin` en [-1, 1]); sin normalizar,
    una red densa entrena mal o no converge.
    """

    def __init__(self, epochs=300, batch_size=16, paciencia=25, verbose=0):
        self.epochs, self.batch_size = epochs, batch_size
        self.paciencia, self.verbose = paciencia, verbose
        self.modelo = self.scaler_X = self.scaler_y = None

    def fit(self, X, y, X_val=None, y_val=None):
        import numpy as np
        from sklearn.preprocessing import StandardScaler
        from tensorflow import keras

        Xv = X.values if hasattr(X, "values") else np.asarray(X)
        yv = np.asarray(y, dtype=float).reshape(-1, 1)
        self.scaler_X = StandardScaler().fit(Xv)
        self.scaler_y = StandardScaler().fit(yv)
        Xs, ys = self.scaler_X.transform(Xv), self.scaler_y.transform(yv).ravel()

        self.modelo = construir_red_neuronal(Xs.shape[1])
        val = None
        if X_val is not None and y_val is not None:
            Xvv = X_val.values if hasattr(X_val, "values") else np.asarray(X_val)
            yvv = np.asarray(y_val, dtype=float).reshape(-1, 1)
            val = (self.scaler_X.transform(Xvv), self.scaler_y.transform(yvv).ravel())

        cb = keras.callbacks.EarlyStopping(monitor="val_loss" if val else "loss",
                                            patience=self.paciencia, restore_best_weights=True)
        self.modelo.fit(Xs, ys, validation_data=val, epochs=self.epochs,
                         batch_size=self.batch_size, callbacks=[cb], verbose=self.verbose)
        return self

    def predict(self, X):
        import numpy as np
        Xv = X.values if hasattr(X, "values") else np.asarray(X)
        Xs = self.scaler_X.transform(Xv)
        pred_s = self.modelo.predict(Xs, verbose=0).ravel()
        return self.scaler_y.inverse_transform(pred_s.reshape(-1, 1)).ravel()

    def guardar(self, prefijo):
        import joblib
        self.modelo.save(f"{prefijo}.keras")
        joblib.dump({"scaler_X": self.scaler_X, "scaler_y": self.scaler_y}, f"{prefijo}_scalers.pkl")

    @classmethod
    def cargar(cls, prefijo):
        import joblib
        from tensorflow import keras
        obj = cls()
        obj.modelo = keras.models.load_model(f"{prefijo}.keras")
        esc = joblib.load(f"{prefijo}_scalers.pkl")
        obj.scaler_X, obj.scaler_y = esc["scaler_X"], esc["scaler_y"]
        return obj


# --------------------------------------------------------------------------- #
# 3. LSTM
# --------------------------------------------------------------------------- #
def preparar_secuencias(df, ventana: int = 8, columnas: list | None = None):
    """Construye secuencias (X:[n,ventana,n_features], y:[n]) para la LSTM.

    Usa una ventana de `ventana` semanas anteriores (precio y, opcionalmente,
    otras variables) para predecir el precio de la semana siguiente. Se arma
    POR PRODUCTO y en orden temporal, para no mezclar series ni filtrar
    información entre productos (igual que los lags de `features.py`).

    Returns: X (ndarray), y (ndarray), fechas (ndarray), productos (ndarray).
    """
    import numpy as np

    columnas = columnas or [config.COL_PRECIO]
    idx_precio = columnas.index(config.COL_PRECIO)
    df = df.sort_values([config.COL_PRODUCTO, config.COL_FECHA])

    X, y, fechas, productos = [], [], [], []
    for prod, sub in df.groupby(config.COL_PRODUCTO):
        vals = sub[columnas].to_numpy(dtype=float)
        fechas_sub = sub[config.COL_FECHA].to_numpy()
        for i in range(ventana, len(sub)):
            X.append(vals[i - ventana:i])
            y.append(vals[i, idx_precio])
            fechas.append(fechas_sub[i])
            productos.append(prod)

    orden = np.argsort(fechas)  # re-ordena por fecha (los grupos se procesaron por producto)
    X, y = np.asarray(X)[orden], np.asarray(y)[orden]
    fechas, productos = np.asarray(fechas)[orden], np.asarray(productos)[orden]
    return X, y, fechas, productos


def construir_lstm(input_shape, unidades: int = 64):
    """Arquitectura de una red LSTM (Keras) para la secuencia de precio."""
    from tensorflow import keras
    modelo = keras.Sequential([
        keras.layers.Input(shape=input_shape),
        keras.layers.LSTM(unidades, return_sequences=False),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(max(unidades // 2, 8), activation="relu"),
        keras.layers.Dense(1),
    ])
    modelo.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return modelo


class LSTMRegresor:
    """LSTM (Keras) con escalado integrado e interfaz fit/predict tipo sklearn."""

    def __init__(self, unidades=64, epochs=300, batch_size=16, paciencia=25, verbose=0):
        self.unidades, self.epochs, self.batch_size = unidades, epochs, batch_size
        self.paciencia, self.verbose = paciencia, verbose
        self.modelo = self.scaler_X = self.scaler_y = None

    def fit(self, X, y, X_val=None, y_val=None):
        import numpy as np
        from sklearn.preprocessing import StandardScaler
        from tensorflow import keras

        n, ventana, nf = X.shape
        self.scaler_X = StandardScaler().fit(X.reshape(-1, nf))
        Xs = self.scaler_X.transform(X.reshape(-1, nf)).reshape(n, ventana, nf)
        yv = np.asarray(y, dtype=float).reshape(-1, 1)
        self.scaler_y = StandardScaler().fit(yv)
        ys = self.scaler_y.transform(yv).ravel()

        self.modelo = construir_lstm((ventana, nf), unidades=self.unidades)
        val = None
        if X_val is not None and y_val is not None:
            nv = X_val.shape[0]
            Xvs = self.scaler_X.transform(X_val.reshape(-1, nf)).reshape(nv, ventana, nf)
            yvv = self.scaler_y.transform(np.asarray(y_val, dtype=float).reshape(-1, 1)).ravel()
            val = (Xvs, yvv)

        cb = keras.callbacks.EarlyStopping(monitor="val_loss" if val else "loss",
                                            patience=self.paciencia, restore_best_weights=True)
        self.modelo.fit(Xs, ys, validation_data=val, epochs=self.epochs,
                         batch_size=self.batch_size, callbacks=[cb], verbose=self.verbose)
        return self

    def predict(self, X):
        import numpy as np
        n, ventana, nf = X.shape
        Xs = self.scaler_X.transform(X.reshape(-1, nf)).reshape(n, ventana, nf)
        pred_s = self.modelo.predict(Xs, verbose=0).ravel()
        return self.scaler_y.inverse_transform(pred_s.reshape(-1, 1)).ravel()

    def guardar(self, prefijo):
        import joblib
        self.modelo.save(f"{prefijo}.keras")
        joblib.dump({"scaler_X": self.scaler_X, "scaler_y": self.scaler_y}, f"{prefijo}_scalers.pkl")

    @classmethod
    def cargar(cls, prefijo):
        import joblib
        from tensorflow import keras
        obj = cls()
        obj.modelo = keras.models.load_model(f"{prefijo}.keras")
        esc = joblib.load(f"{prefijo}_scalers.pkl")
        obj.scaler_X, obj.scaler_y = esc["scaler_X"], esc["scaler_y"]
        return obj


# --------------------------------------------------------------------------- #
# 4. Ensemble
# --------------------------------------------------------------------------- #
class EnsemblePromedio:
    """Ensemble simple: promedio ponderado de las predicciones de varios modelos ya entrenados."""

    def __init__(self, modelos, pesos, nombres):
        import numpy as np
        self.modelos, self.nombres = modelos, nombres
        self.pesos = np.asarray(pesos, dtype=float)

    def predict(self, X):
        import numpy as np
        preds = np.column_stack([m.predict(X) for m in self.modelos])
        return preds @ self.pesos


class EnsembleStacking:
    """Ensemble por stacking: meta-modelo (Ridge) sobre las predicciones de los modelos base."""

    def __init__(self, modelos, meta_modelo, nombres):
        self.modelos, self.meta_modelo, self.nombres = modelos, meta_modelo, nombres

    def predict(self, X):
        import numpy as np
        preds = np.column_stack([m.predict(X) for m in self.modelos])
        return self.meta_modelo.predict(preds)


def construir_ensemble(modelos: dict, X_val, y_val, metodo: str = "stacking"):
    """Combina varios modelos YA ENTRENADOS usando un tramo de validación separado.

    Importante: `X_val`/`y_val` NO deben ser el tramo de test final — deben
    venir de `evaluation.split_train_val_test`, para no filtrar información
    del test en el ajuste del ensemble (pesos o meta-modelo).

    metodo="voting": promedio simple de las predicciones de todos los modelos.
    metodo="stacking": ajusta un meta-modelo Ridge sobre las predicciones de
    los modelos base en `X_val` (predicciones "fuera de muestra" respecto al
    entrenamiento de cada modelo base).
    """
    import numpy as np

    nombres = list(modelos.keys())
    lista_modelos = [modelos[n] for n in nombres]
    preds_val = np.column_stack([m.predict(X_val) for m in lista_modelos])

    if metodo == "voting":
        pesos = np.ones(len(nombres)) / len(nombres)
        return EnsemblePromedio(lista_modelos, pesos, nombres)

    if metodo == "stacking":
        from sklearn.linear_model import Ridge
        meta = Ridge(alpha=1.0, random_state=config.RANDOM_STATE).fit(preds_val, y_val)
        return EnsembleStacking(lista_modelos, meta, nombres)

    raise ValueError('metodo debe ser "voting" o "stacking".')
