import os

# IMPORTANTE: estas variables deben fijarse ANTES de importar numpy/sklearn/xgboost/etc.
# (dentro de `from utils import *`), porque las librerías de álgebra lineal (BLAS/OpenMP)
# leen estas variables una sola vez, al cargarse. Si no se limitan, cada worker de joblib
# además abre su propio pool interno de hilos, y con varios workers en paralelo se termina
# con decenas de hilos compitiendo por CPU/RAM -> esto es una causa muy común de
# TerminatedWorkerError (segfault o memoria agotada) en Windows.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")  # backend sin interfaz grafica: nunca abre ventanas emergentes,
                        # sin importar si el codigo (propio o de utils.py) llama a plt.show()

from utils import *
from joblib import Parallel, delayed
from joblib.externals.loky.process_executor import TerminatedWorkerError
from copy import deepcopy
from sklearn.base import clone

import logging
import json
import time
import gc
from datetime import timedelta


# =========== LOAD DATA =================

df_imputed = pd.read_csv(CFG.path_df_imputed_corrected)
df_imputed.columns = clean_feature_names(df_imputed.columns)


N_ITERATIONS = 20
BASE_SEED = 42

# N_JOBS_MODELS: cuántos modelos se entrenan en paralelo (paralelismo "externo").
# N_JOBS_GRID: n_jobs dentro de cada GridSearchCV (paralelismo "interno").
# CLAVE: nunca pongas ambos > 1 al mismo tiempo. Si N_JOBS_MODELS > 1, cada uno de esos
# procesos YA está ocupando un núcleo completo; si además cada GridSearchCV intenta abrir
# su propio pool de procesos, terminas con N_JOBS_MODELS * N_JOBS_GRID procesos pesados
# simultáneos (más los hilos de BLAS de cada uno) -> sobre-suscripción de CPU/RAM y
# workers muriendo (TerminatedWorkerError). Por eso aquí se deja el interno en 1.
N_JOBS_MODELS = 1
N_JOBS_GRID = 6

# % de outliers a evaluar (0 = baseline, sin modificar). Ajusta a gusto.
OUTLIER_PERCENTAGES = [0.1, 0.2, 0.3]
N_STD_OUTLIERS = 3  # Número de desviaciones estándar para desplazar los outliers

# Carpeta de salida de este experimento
OUTPUT_PATH = f"{CFG.Root}/Resultados/outliers_experiment_Quartiles/"
os.makedirs(OUTPUT_PATH, exist_ok=True)


# ============================================================
# =====================  LOGGER  ==============================
# ============================================================
LOG_DIR = f"{OUTPUT_PATH}logs/"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = f"{LOG_DIR}outliers_experiment.log"

logger = logging.getLogger("outliers_experiment")
logger.setLevel(logging.INFO)
logger.propagate = False

# Evita duplicar handlers si el script/módulo se importa/ejecuta más de una vez
if not logger.handlers:
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


# ============================================================
# =================  CHECKPOINT / RESUME  =====================
# ============================================================
# Guarda qué combinaciones (outlier_percentage, iteration_idx) ya se
# terminaron con éxito. Al reiniciar el script, se saltan automáticamente.
#
# Se protege con un lock basado en archivo (sin dependencias externas) porque
# puede haber más de un proceso queriendo leer/escribir el mismo checkpoint.json
# al mismo tiempo (p. ej. si corres el script varias veces en paralelo, uno por
# cada outlier_percentage, o si en el futuro se agrega checkpoint por modelo
# dentro del Parallel de joblib). Sin el lock, dos procesos escribiendo a la
# vez pueden pisarse la escritura o dejar el JSON corrupto/incompleto.
CHECKPOINT_FILE = f"{OUTPUT_PATH}checkpoint.json"
CHECKPOINT_LOCK_FILE = CHECKPOINT_FILE + ".lock"


class SimpleFileLock:
    """Lock inter-proceso basado en la creación exclusiva de un archivo.

    Funciona igual entre hilos, entre procesos y entre workers de joblib
    (backend 'loky'), y no requiere ninguna librería externa (no depende de
    fcntl, por lo que también funciona en Windows).
    """

    def __init__(self, lock_path, timeout=120, poll_interval=0.05):
        self.lock_path = lock_path
        self.timeout = timeout
        self.poll_interval = poll_interval
        self._fd = None

    def acquire(self):
        start = time.time()
        while True:
            try:
                # O_CREAT | O_EXCL es atómico a nivel de sistema operativo:
                # si el archivo ya existe, falla en vez de sobrescribir.
                self._fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                return
            except FileExistsError:
                if time.time() - start > self.timeout:
                    raise TimeoutError(
                        f"No se pudo adquirir el lock '{self.lock_path}' tras {self.timeout}s. "
                        "Puede que un proceso anterior haya muerto dejando el lock huérfano; "
                        "si estás seguro de que no hay otra ejecución activa, borra ese archivo."
                    )
                time.sleep(self.poll_interval)

    def release(self):
        if self._fd is not None:
            try:
                os.close(self._fd)
            except OSError:
                pass
            try:
                os.remove(self.lock_path)
            except OSError:
                pass
            self._fd = None

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


def _read_checkpoint_raw():
    """Lee el checkpoint tal cual está en disco (sin lock; usar dentro de uno)."""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return set(tuple(x) for x in data.get("completadas", []))
    return set()


def _write_checkpoint_raw(completadas):
    """Escribe el checkpoint de forma atómica (sin lock; usar dentro de uno)."""
    data = {"completadas": [list(x) for x in sorted(completadas)]}
    tmp_file = f"{CHECKPOINT_FILE}.tmp.{os.getpid()}"
    with open(tmp_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_file, CHECKPOINT_FILE)  # rename atómico, evita checkpoints a medio escribir


def load_checkpoint():
    """Carga el set de combinaciones (outlier, iteracion) ya completadas."""
    with SimpleFileLock(CHECKPOINT_LOCK_FILE):
        completadas = _read_checkpoint_raw()

    if completadas:
        logger.info(
            f"Checkpoint encontrado en '{CHECKPOINT_FILE}': "
            f"{len(completadas)} combinaciones ya completadas. Se retomará la ejecución."
        )
    else:
        logger.info("No se encontró checkpoint previo. Se iniciará una ejecución nueva.")
    return completadas


def mark_completed(key):
    """Marca `key` como completada en el checkpoint de forma segura entre procesos.

    Relee el archivo bajo el lock antes de escribir (en vez de confiar en el
    set que tenemos en memoria), para no perder combinaciones que otro
    proceso paralelo haya marcado mientras tanto. Devuelve el set actualizado.
    """
    with SimpleFileLock(CHECKPOINT_LOCK_FILE):
        completadas = _read_checkpoint_raw()
        completadas.add(key)
        _write_checkpoint_raw(completadas)
    return completadas


def outlier_summary_done(iter_summary_dir, output_name):
    """Comprueba si el resumen de iteraciones para este % de outliers ya se generó."""
    return os.path.exists(f"{iter_summary_dir}{output_name}")


def agregar_outliers(X_test, porcentaje=0.02, n_std=N_STD_OUTLIERS, columnas=None,
                      n_columnas_por_fila=1, random_state=42):
    """
    Agrega outliers al conjunto de test, usando la media y desviación
    estándar del propio conjunto de test que se le pasa.
 
    Parameters
    ----------
    X_test : pd.DataFrame
    porcentaje : float
        Fracción de filas a modificar (ej. 0.2 = 20%).
    n_std : int o float
        Número de desviaciones estándar para desplazar el valor (+/-).
    columnas : list o None
        Columnas donde insertar outliers. Si es None, usa todas las numéricas.
    n_columnas_por_fila : int
        Número de variables que se modificarán en cada fila seleccionada.
    random_state : int
 
    Returns
    -------
    X_test_outliers : pd.DataFrame
    filas_modificadas : np.ndarray (índices de las filas modificadas)
    """
    rng = np.random.default_rng(random_state)  # Generador de números aleatorios
 
    X_test_out = X_test.copy()
 
    if columnas is None:
        columnas = X_test.select_dtypes(include=np.number).columns.tolist()
 
    medias = X_test[columnas].mean()
    desv = X_test[columnas].std()
 
    n_filas = max(1, int(len(X_test_out) * porcentaje)) if porcentaje > 0 else 0
 
    if n_filas == 0:
        return X_test_out, np.array([])
    # escoger aleatoriamente las filas a modificar
    filas = rng.choice(X_test_out.index, size=n_filas, replace=False)
 
    for fila in filas:
        # escoger aleatoriamente las columnas a modificar
        cols = rng.choice(columnas, size=min(n_columnas_por_fila, len(columnas)), replace=False)
        for col in cols:
            signo = rng.choice([-1, 1])
            X_test_out.loc[fila, col] = medias[col] + signo * n_std * desv[col]
 
    return X_test_out, filas


def train_test_model_outliers(df_imputed, n_clases, model_name, model_config, element="Quartiles",
                              usar_smote=True, mostrar_graficos=True, calcular_shap=True,
                              h5_file=None,
                              dir_path="../",
                              best_variables=None, train_pca=False, n_components=None,
                              CFG=None,
                              seed=None, outlier=0.0):
    """Función principal para entrenar y evaluar un modelo.
    Utiliza Crossvalidación en GridSearch.

    Args:
        df_imputed (DataFrame): DataFrame con datos imputados.
        n_clases (int): Número de clases para la codificación.
        model_name (str): Nombre del modelo.
        model_config (dict): Configuración del modelo (estimator y param_grid).
        element (str): Elemento a utilizar para la codificación individual.
        usar_smote (bool): Si se debe usar SMOTE para balancear clases.
        mostrar_graficos (bool): Si se deben mostrar gráficos de confusión.
        calcular_shap (bool): Si se deben calcular valores SHAP.
        h5_file: Archivo HDF5 abierto para guardar resultados.
        dir_path (str): Ruta para almacenar los modelos de cada algoritmo
                        en formato binario .pkl
        train_pca (bool): Opción para entrenar con variables reduciar por PCA
        n_components(int): Number of components when applying PCA.
    Returns:
        dict: Resultados del entrenamiento y evaluación del modelo.
    """
    if seed is None:
        seed = 42

    logger.info(f"[{model_name}] Iniciando entrenamiento (seed={seed}, outlier={outlier}).")

    # Preparar datos
    X_train, X_test, y_train, y_test, feature_names, class_dist = preparar_datos(
        df_imputed,
        n_clases,
        element=element,
        random_state=seed,
        best_variables=best_variables,
        CFG=CFG
    )

    if train_pca:
        # Aplicar PCA
        X_train, pca = calcuate_PCA(X_train, n_components=n_components)
        X_test = pca.transform(X_test)

    # Construir objeto de pipeline
    pipe = build_pipeline(model_config=model_config,
                          usar_smote=usar_smote,
                          seed=seed)

    # Configurar KFolds Estratificados
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    # Construir Grid Search con CV
    grid = GridSearchCV(pipe, model_config['param_grid'], cv=cv,
                       scoring='f1_micro', n_jobs=N_JOBS_GRID, verbose=2,
                       return_train_score=True)
    grid.fit(X_train, y_train)

    logger.info(f"[{model_name}] GridSearch finalizado. Mejores parámetros: {grid.best_params_}")

    # Métricas train del mejor modelo (usando cross-validation)
    nested_score = cross_validate(
        grid.best_estimator_, X=X_train, y=y_train,
        cv=cv, scoring=['f1_micro', 'f1_macro', 'accuracy'],
        return_train_score=True
    )
    acc_train = np.mean(nested_score['train_accuracy'])
    f1_train = np.mean(nested_score['train_f1_micro'])
    f1_train_macro = np.mean(nested_score['train_f1_macro'])
    # ================ TEST =====================
    X_test, _ = agregar_outliers(X_test, porcentaje=outlier, n_std=N_STD_OUTLIERS, random_state=seed)

    y_test_pred = grid.predict(X_test)

    # Métricas test
    acc_test, f1_test, f1_test_macro = return_classification_metrics(
        y_test, y_test_pred
    )

    # =================== Classification Report =============
    class_report = classification_report(y_test, y_test_pred)
    os.makedirs(f"{dir_path}/results", exist_ok=True)
    with open(f"{dir_path}/results/{model_name.replace(' ', '_')}_classification_report_{element}.txt", "w") as f:
        f.write(class_report)
    logger.info(f"[{model_name}] Classification report (test):\n{class_report}")
    class_report_dict = classification_report(y_test, y_test_pred, output_dict=True)

    # Matrices de confusión
    cm_test = confusion_matrix(y_test, y_test_pred)

    #Guardar modelo
    #model_path = f"../Resultados/classification/models/{model_name.replace(' ', '_')}_nclases_{n_clases}.pkl"
    #Revisar si el directorio existe

    if CFG.individual_train:
        model_path =  f"{dir_path}/models/{model_name.replace(' ', '_')}_nclases_{3}_{element}.pkl"
    elif CFG.cuartiles_train:
        model_path =  f"{dir_path}/models/{model_name.replace(' ', '_')}_nclases_{2}_cuartiles.pkl"
    else:
        model_path = f"{dir_path}/models/{model_name.replace(' ', '_')}_nclases_{n_clases}.pkl"
    os.makedirs(f"{dir_path}/models", exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(grid.best_estimator_, f)

    # Imprimir resultados (función externa de utils, mantiene su propio print/log interno)
    print_classification_report(model_name, n_clases, acc_train, acc_test, f1_train, f1_test, grid.best_params_, class_dist)

    # # Gráficos de confusión
    # fig_cm_train = plot_confusion_matrix(cm_train, classes=np.unique(y_train),
    #                     title=f"{model_name} Train - {n_clases} classes")
    if mostrar_graficos:
        plt.show()

    fig_cm_test = plot_confusion_matrix(cm_test, classes=np.unique(y_test),
                        title=f"{model_name} Test - {n_clases} classes")

    # Con backend "Agg" no se abre ninguna ventana; aun asi, solo llamamos a
    # plt.show() si se pide explicitamente. En su lugar, guardamos la figura
    # en disco (que es lo que realmente se necesita) y liberamos memoria.
    if mostrar_graficos:
        plt.show()

    if fig_cm_test is not None:
        cm_fig_path = f"{dir_path}/results/{model_name.replace(' ', '_')}_confusion_matrix_test_{element}.png"
        try:
            fig_cm_test.savefig(cm_fig_path, dpi=150, bbox_inches="tight")
            logger.info(f"[{model_name}] Matriz de confusión (test) guardada en: {cm_fig_path}")
        except Exception:
            logger.exception(f"[{model_name}] No se pudo guardar la matriz de confusión en disco.")
        finally:
            plt.close(fig_cm_test)
    else:
        plt.close("all")

    # ================ Calcular SHAP ===========================
    shap_values = None
    fig_shap = None
    X_scaled_df = None
    if calcular_shap:
        shap_values, X_scaled_df = calculate_shap(grid, X_test,
                                                dir_path, model_name,
                                                feature_names, model_config,
                                                n_clases,
                                                mostrar_graficos)

    
    # ============ Almacenar Resultados ===========================
    if CFG.individual_train:
        n_clases_str = f"{3}_{element}"
    elif CFG.cuartiles_train:
        n_clases_str = "2_Quartiles"
    else:
        n_clases_str = str(n_clases)
    resultados = {
        'y_true': y_test.tolist(),
        'y_pred': y_test_pred.tolist(),
        'n_clases': n_clases_str,
        'model_name': model_name,
        'accuracy_train': acc_train,
        'accuracy_test': acc_test,
        'f1_train': f1_train,
        'f1_train_macro': f1_train_macro,
        'f1_test': f1_test,
        'f1_macro_test': f1_test_macro,
        'best_params': grid.best_params_,
        'class_distribution': class_dist,
        'classification_report': class_report_dict,
        'confusion_matrix_test': cm_test,
        'grid_search': grid,
        'shap_values': shap_values,
        'X_scaled_df': X_test if X_scaled_df is None else X_scaled_df,
        'feature_names': feature_names,
    }

    plt.close("all")  # asegura que no queden figuras abiertas entre entrenamientos

    logger.info(f"[{model_name}] Entrenamiento completo. acc_test={acc_test:.4f} | f1_test={f1_test:.4f}")

    return resultados

def get_models_config_for_seed(seed):
    """Retorna una copia de la configuracion de modelos usando una semilla indicada para su inicializacion."""
    models_config_seed = {}

    for model_name, model_cfg in MODELS_CONFIG.items():
        cfg_copy = deepcopy(model_cfg)
        estimator = clone(model_cfg['estimator'])

        estimator_params = estimator.get_params(deep=False)
        seed_params = {}
        if 'random_state' in estimator_params:
            seed_params['random_state'] = seed
        if 'seed' in estimator_params:
            seed_params['seed'] = seed

        if seed_params:
            estimator.set_params(**seed_params)

        cfg_copy['estimator'] = estimator
        models_config_seed[model_name] = cfg_copy

    return models_config_seed



def entrenar_modelo_cuartiles(model_name, model_config, df_imputed, class_path, seed, outlier=0.0):
    """Entrena un modelo para clasificacion por cuartiles."""
    # joblib workers can start with config defaults; enforce cuartiles mode here.
    CFG.individual_train = False
    CFG.cuartiles_train = True
    CFG.include_prod = False

    dir_path = f"{class_path}{model_name.replace(' ', '_')}/"
    os.makedirs(dir_path, exist_ok=True)

    resultado = train_test_model_outliers(
        df_imputed=df_imputed,
        n_clases=None,
        model_name=model_name,
        model_config=model_config,
        element=None,
        usar_smote=False,
        mostrar_graficos=False,
        calcular_shap=False,
        dir_path=dir_path,
        CFG=CFG,
        seed=seed,
        outlier=outlier
    )
    return (model_name, resultado)



def _entrenar_todos_los_modelos(models_config, df_imputed, class_path, seed, outlier, n_jobs):
    """Corre entrenar_modelo_cuartiles para todos los modelos con el n_jobs indicado."""
    return Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(entrenar_modelo_cuartiles)(
            model_name=model_name,
            model_config=model_config,
            df_imputed=df_imputed,
            class_path=class_path,
            seed=seed,
            outlier=outlier
        )
        for model_name, model_config in models_config.items()
    )


def run_non_nested_iteration(df_imputed, models_config, class_path, seed, outlier=0.0):
    CFG.class_path = class_path
    os.makedirs(CFG.class_path, exist_ok=True)
    CFG.individual_train = False
    CFG.cuartiles_train = True
    CFG.include_prod = False
    CFG.path_pkl_results_classification = f"{CFG.class_path}class_models_cuartiles_all_models.pkl"

    try:
        all_results_list = _entrenar_todos_los_modelos(
            models_config, df_imputed, class_path, seed, outlier, n_jobs=N_JOBS_MODELS
        )
    except TerminatedWorkerError:
        # Un worker murió (segfault / memoria agotada). En vez de perder toda la
        # iteración, reintentamos una sola vez en modo totalmente secuencial
        # (n_jobs=1): más lento, pero evita la sobre-suscripción de CPU/RAM que
        # suele causar este error. Si vuelve a fallar aquí, el problema no es de
        # paralelismo sino de un modelo/config específico (o RAM insuficiente).
        logger.error(
            "Un worker de joblib murió inesperadamente durante el entrenamiento en paralelo "
            "(posible falta de memoria o sobre-suscripción de hilos). "
            "Reintentando esta iteración de forma secuencial (n_jobs=1)..."
        )
        gc.collect()
        all_results_list = _entrenar_todos_los_modelos(
            models_config, df_imputed, class_path, seed, outlier, n_jobs=1
        )
        logger.info("Reintento secuencial exitoso.")

    all_results = {}
    for model_name, resultado in all_results_list:
        if model_name not in all_results:
            all_results[model_name] = []
        all_results[model_name].append(resultado)

    logger.info(f"[CUARTILES NO NESTED] Total de combinaciones: {len(all_results_list)}")

    with open(CFG.path_pkl_results_classification, 'wb') as pkl_file:
        pickle.dump(all_results, pkl_file)

    compare_classification_models(all_results, CFG=CFG)
    save_results_general(all_results, CFG.class_path)

def run_iteration_outlier(iteration_idx, seed, outlier):
    logger.info(f"Iteracion {iteration_idx}/{N_ITERATIONS} | seed={seed} | outlier={outlier}")
    models_config_seed = get_models_config_for_seed(seed)
    
    outlier_str = f"{int(outlier * 100)}" if outlier > 0 else "0"
    
    # preparar ruta
    outlier_dir = f"{OUTPUT_PATH}outlier_{outlier_str}/"   
    os.makedirs(outlier_dir, exist_ok=True)

    class_path_non_nested = (
        f"{outlier_dir}classification_cuartiles_exclude_prod/"
        f"iter_{iteration_idx:02d}_seed_{seed}/"
    )
    run_non_nested_iteration(df_imputed, models_config_seed, class_path_non_nested, seed, outlier=outlier)

def _get_column_name(df, options):
    for col in options:
        if col in df.columns:
            return col
    return None

def _best_model_and_accuracy(df_results):
    """Obtiene el mejor modelo usando la mayor Accuracy_Test."""
    model_col = _get_column_name(df_results, ['Model', 'model_name'])
    acc_col = _get_column_name(df_results, ['Accuracy_Test', 'accuracy_test'])

    if model_col is None or acc_col is None:
        return None, None

    df_valid = df_results.copy()
    df_valid[acc_col] = pd.to_numeric(df_valid[acc_col], errors='coerce')
    df_valid = df_valid.dropna(subset=[acc_col])

    if df_valid.empty:
        return None, None

    best_row = df_valid.loc[df_valid[acc_col].idxmax()]
    return best_row[model_col], float(best_row[acc_col])

def build_cuartiles_iterations_summary(base_path, n_iterations, base_seed=42, output_name='resumen_metricas_cuartiles_nested_iteraciones.csv',
                                       iter_summary_dir = f"{CFG.Root}/train/train_iterations/iter_summary/"):
    """Resume mejor modelo y accuracy por iteracion para experimento de cuartiles."""
    rows = []

    for iteration_idx in range(1, n_iterations + 1):
        seed = base_seed + iteration_idx - 1
        csv_path = (
            f"{base_path}iter_{iteration_idx:02d}_seed_{seed}/"
            "resultados_modelos_completos.csv"
        )

        row = {
            'iteraciones': iteration_idx,
            'accuracy': None,
            'Best Model': None,
        }

        if os.path.exists(csv_path):
            df_results = pd.read_csv(csv_path)
            best_model, best_acc = _best_model_and_accuracy(df_results)
            row['accuracy'] = best_acc
            row['Best Model'] = best_model

        rows.append(row)

    df_summary = pd.DataFrame(rows)
    
    os.makedirs(iter_summary_dir, exist_ok=True)
    output_path = f"{iter_summary_dir}{output_name}"
    df_summary.to_csv(output_path, index=False)
    logger.info(f"Resumen de iteraciones guardado en: {output_path}")

    return df_summary


# ============================================================
# =====================  MAIN LOOP  ============================
# ============================================================
if __name__ == "__main__":

    completadas = load_checkpoint()
    inicio_total = time.time()

    try:
        for outlier_percentage in OUTLIER_PERCENTAGES:
            for iteration_idx in range(1, N_ITERATIONS + 1):
                key = (outlier_percentage, iteration_idx)

                if key in completadas:
                    logger.info(
                        f"Saltando outlier={outlier_percentage} | iteracion={iteration_idx} "
                        f"(ya completada en una ejecución anterior)."
                    )
                    continue

                seed = BASE_SEED + iteration_idx - 1
                t0 = time.time()

                # NOTE: esta funcion cambia respecto a NPK en la ruta de guardado
                run_iteration_outlier(iteration_idx, seed, outlier_percentage)

                elapsed = timedelta(seconds=int(time.time() - t0))
                logger.info(
                    f"Completada outlier={outlier_percentage} | iteracion={iteration_idx} "
                    f"en {elapsed}."
                )

                # Se marca como completada y se persiste el checkpoint de inmediato
                # (bajo lock, releyendo el archivo para no pisar el progreso de otro
                # proceso), así una interrupción justo después no pierde nada.
                completadas = mark_completed(key)

            # ---- Resumen de iteraciones para este % de outliers ----
            outlier_str = f"{int(outlier_percentage * 100)}" if outlier_percentage > 0 else "0"
            outlier_dir = f"{OUTPUT_PATH}outlier_{outlier_str}/"
            class_path_non_nested = f"{outlier_dir}classification_cuartiles_exclude_prod/"
            iter_summary_dir = f"{outlier_dir}iter_summary/"
            output_name = "resumen_metricas_cuartiles_non_nested_iteraciones.csv"

            build_cuartiles_iterations_summary(
                base_path=class_path_non_nested,
                n_iterations=N_ITERATIONS,
                base_seed=BASE_SEED,
                output_name=output_name,
                iter_summary_dir=iter_summary_dir
            )

        elapsed_total = timedelta(seconds=int(time.time() - inicio_total))
        logger.info(f"Ejecución completa. Tiempo total: {elapsed_total}.")

    except KeyboardInterrupt:
        # No hace falta volver a guardar aquí: cada combinación ya se persistió
        # (bajo lock) justo al terminar. Si la interrupción llega a mitad de una
        # iteración, esa iteración simplemente se reintentará en la próxima corrida.
        logger.warning(
            "Ejecución interrumpida manualmente (Ctrl+C). "
            f"Progreso guardado en '{CHECKPOINT_FILE}'. "
            "Vuelve a correr el script para retomar donde quedó."
        )
        raise

    except Exception:
        logger.exception(
            "Error inesperado durante la ejecución. "
            f"Progreso guardado en '{CHECKPOINT_FILE}'. "
            "Corrige el error y vuelve a correr el script para retomar."
        )
        raise