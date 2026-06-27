from utils import *
from joblib import Parallel, delayed
from copy import deepcopy
from sklearn.base import clone


# =========== LOAD DATA =================

df_imputed = pd.read_csv(CFG.path_df_imputed_corrected)
df_imputed.columns = clean_feature_names(df_imputed.columns)


N_ITERATIONS = 20
BASE_SEED = 42
N_THREADS = 6


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

    resultado = train_test_model(
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
            class_path=CFG.class_path,
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


def entrenar_modelo_cuartiles_nested(model_name, model_config, df_imputed, class_path, seed):
    """Entrena un modelo nested para clasificacion por cuartiles."""
    # joblib workers can start with config defaults; enforce cuartiles mode here.
    CFG.individual_train = False
    CFG.cuartiles_train = True
    CFG.include_prod = False

    dir_path = f"{class_path}{model_name.replace(' ', '_')}/"
    os.makedirs(dir_path, exist_ok=True)

    resultado = train_test_class_nested(
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


def run_nested_iteration(df_imputed, models_config, class_path, seed):
    CFG.class_path = class_path
    os.makedirs(CFG.class_path, exist_ok=True)
    CFG.individual_train = False
    CFG.cuartiles_train = True
    CFG.include_prod = False
    CFG.path_pkl_results_classification = f"{CFG.class_path}class_models_cuartiles_all_models.pkl"

    all_results_list = Parallel(n_jobs=N_THREADS, verbose=10)(
        delayed(entrenar_modelo_cuartiles_nested)(
            model_name=model_name,
            model_config=model_config,
            df_imputed=df_imputed,
            class_path=CFG.class_path,
            seed=seed
        )
        for model_name, model_config in models_config.items()
    )

    all_results = {}
    for model_name, resultado in all_results_list:
        if model_name not in all_results:
            all_results[model_name] = []
        all_results[model_name].append(resultado)

    print(f"[CUARTILES NESTED] Total de combinaciones: {len(all_results_list)}")

    with open(CFG.path_pkl_results_classification, 'wb') as pkl_file:
        pickle.dump(all_results, pkl_file)

    compare_classification_models(all_results, CFG=CFG)
    save_results_general(all_results, CFG.class_path)


def run_iteration(iteration_idx, seed):
    print(f"\nIteracion {iteration_idx}/{N_ITERATIONS} | seed={seed}")
    models_config_seed = get_models_config_for_seed(seed)

    class_path_non_nested = (
        f"{CFG.Root}/Resultados/classification_cuartiles_exclude_prod/"
        f"iter_{iteration_idx:02d}_seed_{seed}/"
    )
    run_non_nested_iteration(df_imputed, models_config_seed, class_path_non_nested, seed)

    class_path_nested = (
        f"{CFG.Root}/Resultados/classification_cuartiles_exclude_prod_nested/"
        f"iter_{iteration_idx:02d}_seed_{seed}/"
    )
    run_nested_iteration(df_imputed, models_config_seed, class_path_nested, seed)


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


def build_cuartiles_iterations_summary(base_path, n_iterations, base_seed=42, output_name='resumen_metricas_cuartiles_nested_iteraciones.csv'):
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
    iter_summary_dir = f"{CFG.Root}/train/train_iterations/iter_summary/"
    os.makedirs(iter_summary_dir, exist_ok=True)
    output_path = f"{iter_summary_dir}{output_name}"
    df_summary.to_csv(output_path, index=False)
    print(f"Resumen de iteraciones guardado en: {output_path}")

    return df_summary


for iteration_idx in range(1, N_ITERATIONS + 1):
    seed = BASE_SEED + iteration_idx - 1
    run_iteration(iteration_idx, seed)


build_cuartiles_iterations_summary(
    base_path=f"{CFG.Root}/Resultados/classification_cuartiles_exclude_prod_nested/",
    n_iterations=N_ITERATIONS,
    base_seed=BASE_SEED,
    output_name='resumen_metricas_cuartiles_nested_iteraciones.csv'
)

build_cuartiles_iterations_summary(
    base_path=f"{CFG.Root}/Resultados/classification_cuartiles_exclude_prod/",
    n_iterations=N_ITERATIONS,
    base_seed=BASE_SEED,
    output_name='resumen_metricas_cuartiles_non_nested_iteraciones.csv'
)

    
