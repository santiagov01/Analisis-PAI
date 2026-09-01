from utils import *
from joblib import Parallel, delayed
from copy import deepcopy
from sklearn.base import clone


# =========== LOAD DATA =================

df_imputed = pd.read_csv(CFG.path_df_imputed_corrected)
df_imputed.columns = clean_feature_names(df_imputed.columns)


N_ITERATIONS = 20
BASE_SEED = 42
N_THREADS = 3

# % de outliers a evaluar (0 = baseline, sin modificar). Ajusta a gusto.
OUTLIER_PERCENTAGES = [0.1, 0.2, 0.3]
N_STD_OUTLIERS = 3  # Número de desviaciones estándar para desplazar los outliers

# Carpeta de salida de este experimento
OUTPUT_PATH = f"{CFG.Root}/Resultados/outliers_experiment_Quartiles/"
os.makedirs(OUTPUT_PATH, exist_ok=True)

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
    rng = np.random.default_rng(random_state) # Generador de números aleatorios
 
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


def train_test_model_outliers(df_imputed, n_clases, model_name, model_config, element = "Quartiles",
                              usar_smote=True, mostrar_graficos=True, calcular_shap=True,
                              h5_file=None,
                              dir_path= "../",
                              best_variables = None, train_pca = False, n_components = None,
                              CFG=None,
                              seed=None):
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
                       scoring='f1_micro', n_jobs=N_THREADS, verbose=2,
                       return_train_score=True)
    grid.fit(X_train, y_train)

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
    X_test, _ = agregar_outliers(X_test, porcentaje=0.2, n_std=N_STD_OUTLIERS, random_state=seed)

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
    print(class_report)
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

    # Imprimir resultados
    print_classification_report(model_name, n_clases, acc_train, acc_test, f1_train, f1_test, grid.best_params_, class_dist)

    # # Gráficos de confusión
    # fig_cm_train = plot_confusion_matrix(cm_train, classes=np.unique(y_train),
    #                     title=f"{model_name} Train - {n_clases} classes")
    if mostrar_graficos:
        plt.show()

    fig_cm_test = plot_confusion_matrix(cm_test, classes=np.unique(y_test),
                        title=f"{model_name} Test - {n_clases} classes")
    if mostrar_graficos:
        plt.show()

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


def entrenar_modelo_cuartiles(model_name, model_config, df_imputed, class_path, seed):
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
        seed=seed
    )
    return (model_name, resultado)


def run_non_nested_iteration(df_imputed, models_config, class_path, seed):
    CFG.class_path = class_path
    os.makedirs(CFG.class_path, exist_ok=True)
    CFG.individual_train = False
    CFG.cuartiles_train = True
    CFG.include_prod = False
    CFG.path_pkl_results_classification = f"{CFG.class_path}class_models_cuartiles_all_models.pkl"

    all_results_list = Parallel(n_jobs=N_THREADS, verbose=10)(
        delayed(entrenar_modelo_cuartiles)(
            model_name=model_name,
            model_config=model_config,
            df_imputed=df_imputed,
            class_path= class_path,
            seed=seed
        )
        for model_name, model_config in models_config.items()
    )

    all_results = {}
    for model_name, resultado in all_results_list:
        if model_name not in all_results:
            all_results[model_name] = []
        all_results[model_name].append(resultado)

    print(f"[CUARTILES NO NESTED] Total de combinaciones: {len(all_results_list)}")

    with open(CFG.path_pkl_results_classification, 'wb') as pkl_file:
        pickle.dump(all_results, pkl_file)

    compare_classification_models(all_results, CFG=CFG)
    save_results_general(all_results, CFG.class_path)

def run_iteration_outlier(iteration_idx, seed, outlier):
    print(f"\nIteracion {iteration_idx}/{N_ITERATIONS} | seed={seed}")
    models_config_seed = get_models_config_for_seed(seed)
    
    outlier_str = f"{int(outlier * 100)}" if outlier > 0 else "0"
    
    # preparar ruta
    outlier_dir = f"{OUTPUT_PATH}outlier_{outlier_str}/"   
    os.makedirs(outlier_dir, exist_ok=True)

    class_path_non_nested = (
        f"{outlier_dir}classification_cuartiles_exclude_prod/"
        f"iter_{iteration_idx:02d}_seed_{seed}/"
    )
    run_non_nested_iteration(df_imputed, models_config_seed, class_path_non_nested, seed)

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
    print(f"Resumen de iteraciones guardado en: {output_path}")

    return df_summary


for outlier_percentage in OUTLIER_PERCENTAGES:
    for iteration_idx in range(1, N_ITERATIONS + 1):
        seed = BASE_SEED + iteration_idx - 1
        #NOTE: esta funcion cambia respecto a NPK en la ruta de guardado
        run_iteration_outlier(iteration_idx, seed, outlier_percentage)
        # preparar ruta
    outlier_str = f"{int(outlier_percentage * 100)}" if outlier_percentage > 0 else "0"
    outlier_dir = f"{OUTPUT_PATH}outlier_{outlier_str}/"
    # Ruta donde estan los resultados de cada iteracion para este porcentaje de outliers
    class_path_non_nested = (
        f"{outlier_dir}classification_cuartiles_exclude_prod/" )
    # Ruta para guardar el resumen de iteraciones
    iter_summary_dir = f"{outlier_dir}iter_summary/"
    # Construir resumen de iteraciones para este porcentaje de outliers
    build_cuartiles_iterations_summary(
        base_path=class_path_non_nested,
        n_iterations=N_ITERATIONS,
        base_seed=BASE_SEED,
        output_name="resumen_metricas_cuartiles_non_nested_iteraciones.csv",
        iter_summary_dir=iter_summary_dir
    )
    
