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


def entrenar_modelo_por_elemento(model_name, model_config, df_imputed, element, class_path, seed):
    """Entrena un modelo para un elemento específico."""
    # joblib workers can start with config defaults; enforce individual mode here.
    CFG.individual_train = True
    CFG.cuartiles_train = False
    CFG.include_prod = False

    dir_path = f"{class_path}{model_name.replace(' ', '_')}/"
    os.makedirs(dir_path, exist_ok=True)

    # NOTE: It can be changed by 'train_test_nested_model'
    resultado = train_test_model(
        df_imputed=df_imputed,
        n_clases=None,  # No utilizar para este experimento
        model_name=model_name,
        model_config=model_config,
        element=element,
        usar_smote=False,
        mostrar_graficos=False,
        calcular_shap=False,
        dir_path=dir_path,
        CFG=CFG,
        seed=seed
    )
    return (model_name, element, resultado)
 
 
def run_non_nested_iteration(df_imputed, models_config, class_path, seed):
    CFG.class_path = class_path
    os.makedirs(CFG.class_path, exist_ok=True)
    CFG.path_pkl_results_classification = f"{CFG.class_path}class_results_individual_elements.pkl"
    CFG.include_prod = False
    CFG.individual_train = True
    CFG.cuartiles_train = False

    all_results_list = Parallel(n_jobs=N_THREADS, verbose=10)(
        delayed(entrenar_modelo_por_elemento)(
            model_name=model_name,
            model_config=model_config,
            df_imputed=df_imputed,
            element=element,
            class_path=CFG.class_path,
            seed=seed
        )
        for model_name, model_config in models_config.items()
        for element in CFG.elements_list
    )

    all_results = {}
    for model_name, element, resultado in all_results_list:
        if model_name not in all_results:
            all_results[model_name] = []
        all_results[model_name].append(resultado)

    print(f"[NO NESTED] Total de combinaciones: {len(all_results_list)}")

    with open(CFG.path_pkl_results_classification, 'wb') as pkl_file:
        pickle.dump(all_results, pkl_file)

    compare_classification_models(all_results, CFG=CFG)
    save_results_general(all_results, CFG.class_path)



def entrenar_modelo_por_elemento_nested(model_name, model_config, df_imputed, element, class_path, seed):
    """Entrena un modelo para un elemento específico."""
    # joblib workers can start with config defaults; enforce individual mode here.
    CFG.individual_train = True
    CFG.cuartiles_train = False
    CFG.include_prod = False

    dir_path = f"{class_path}{model_name.replace(' ', '_')}/"
    os.makedirs(dir_path, exist_ok=True)
    
    resultado = train_test_class_nested(
        df_imputed=df_imputed,
        n_clases=None,
        model_name=model_name,
        model_config=model_config,
        element=element,
        usar_smote=False,
        mostrar_graficos=False,
        calcular_shap=False,
        dir_path=dir_path,
        CFG=CFG,
        seed=seed
    )
    return (model_name, element, resultado)

def run_nested_iteration(df_imputed, models_config, class_path, seed):
    CFG.class_path = class_path
    os.makedirs(CFG.class_path, exist_ok=True)
    CFG.path_pkl_results_classification = f"{CFG.class_path}class_results_individual_elements.pkl"
    CFG.include_prod = False
    CFG.individual_train = True
    CFG.cuartiles_train = False

    all_results_list = Parallel(n_jobs=N_THREADS, verbose=10)(
        delayed(entrenar_modelo_por_elemento_nested)(
            model_name=model_name,
            model_config=model_config,
            df_imputed=df_imputed,
            element=element,
            class_path=CFG.class_path,
            seed=seed
        )
        for model_name, model_config in models_config.items()
        for element in CFG.elements_list
    )

    all_results = {}
    for model_name, element, resultado in all_results_list:
        if model_name not in all_results:
            all_results[model_name] = []
        all_results[model_name].append(resultado)

    print(f"[NESTED] Total de combinaciones: {len(all_results_list)}")

    with open(CFG.path_pkl_results_classification, 'wb') as pkl_file:
        pickle.dump(all_results, pkl_file)

    compare_classification_models(all_results, CFG=CFG)
    save_results_general(all_results, CFG.class_path)



def run_iteration(iteration_idx, seed):
    print(f"\nIteracion {iteration_idx}/{N_ITERATIONS} | seed={seed} ")
    models_config_seed = get_models_config_for_seed(seed)

    class_path_non_nested = (
        f"{CFG.Root}/Resultados/classification_exclude_prod/"
        f"iter_{iteration_idx:02d}_seed_{seed}/"
    )
    run_non_nested_iteration(df_imputed, models_config_seed, class_path_non_nested, seed)

    class_path_nested = (
        f"{CFG.Root}/Resultados/classification_exclude_prod_nested/"
        f"iter_{iteration_idx:02d}_seed_{seed}/"
    )
    run_nested_iteration(df_imputed, models_config_seed, class_path_nested, seed)


def _get_column_name(df, options):
    for col in options:
        if col in df.columns:
            return col
    return None


def _best_model_and_accuracy_by_element(df_results, element_name):
    """Obtiene el mejor modelo por elemento usando la mayor Accuracy_Test."""
    class_col = _get_column_name(df_results, ['N_Classes', 'n_clases'])
    model_col = _get_column_name(df_results, ['Model', 'model_name'])
    acc_col = _get_column_name(df_results, ['Accuracy_Test', 'accuracy_test'])

    if class_col is None or model_col is None or acc_col is None:
        return None, None

    df_element = df_results[
        df_results[class_col].astype(str).str.contains(element_name, case=False, na=False)
    ].copy()

    if df_element.empty:
        return None, None

    df_element[acc_col] = pd.to_numeric(df_element[acc_col], errors='coerce')
    df_element = df_element.dropna(subset=[acc_col])

    if df_element.empty:
        return None, None

    best_row = df_element.loc[df_element[acc_col].idxmax()]
    return best_row[model_col], float(best_row[acc_col])


def build_nested_iterations_summary(base_path, n_iterations, base_seed=42, output_name="resumen_metricas_nested_iteraciones.csv"):
    """Resume el mejor modelo y accuracy de N, P y K para cada iteracion."""
    rows = []

    for iteration_idx in range(1, n_iterations + 1):
        seed = base_seed + iteration_idx - 1
        csv_path = (
            f"{base_path}iter_{iteration_idx:02d}_seed_{seed}/"
            "resultados_modelos_completos.csv"
        )

        row = {
            'Iteracion': iteration_idx,
            'Accuracy N': None,
            'Best Model N': None,
            'Accuracy P': None,
            'Best Model P': None,
            'Accuracy K': None,
            'Best Model K': None,
        }

        if os.path.exists(csv_path):
            df_results = pd.read_csv(csv_path)

            best_n_model, best_n_acc = _best_model_and_accuracy_by_element(df_results, 'Nitrogen')
            best_p_model, best_p_acc = _best_model_and_accuracy_by_element(df_results, 'Phosphorus')
            best_k_model, best_k_acc = _best_model_and_accuracy_by_element(df_results, 'Potassium')

            row['Accuracy N'] = best_n_acc
            row['Best Model N'] = best_n_model
            row['Accuracy P'] = best_p_acc
            row['Best Model P'] = best_p_model
            row['Accuracy K'] = best_k_acc
            row['Best Model K'] = best_k_model

        rows.append(row)

    df_summary = pd.DataFrame(rows)
    # hacer directorio para guardar metricas de iteraciones si no existe
    iter_summary_dir = f"{CFG.Root}/train/train_iterations/iter_summary/"
    os.makedirs(iter_summary_dir, exist_ok=True)
    output_path = f"{iter_summary_dir}{output_name}"
    df_summary.to_csv(output_path, index=False)
    print(f"Resumen de iteraciones (nested) guardado en: {output_path}")

    return df_summary


for iteration_idx in range(1, N_ITERATIONS + 1):
    seed = BASE_SEED + iteration_idx - 1
    run_iteration(iteration_idx, seed)


build_nested_iterations_summary(
    base_path=f"{CFG.Root}/Resultados/classification_exclude_prod_nested/",
    n_iterations=N_ITERATIONS,
    base_seed=BASE_SEED,
    output_name="resumen_metricas_nested_iteraciones.csv"
)

build_nested_iterations_summary(
    base_path=f"{CFG.Root}/Resultados/classification_exclude_prod/",
    n_iterations=N_ITERATIONS,
    base_seed=BASE_SEED,
    output_name="resumen_metricas_non_nested_iteraciones.csv"
)

