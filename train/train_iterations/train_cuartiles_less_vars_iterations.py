from utils import *
from joblib import Parallel, delayed
from copy import deepcopy
from sklearn.base import clone

# =========== LOAD DATA =================

df_imputed = pd.read_csv(CFG.path_df_imputed_corrected)
df_imputed.columns = clean_feature_names(df_imputed.columns)

# ============== READ BEST VARS ===============================
json_best_variables = f"{CFG.Root}/Resultados/classification_exclude_prod/most_frequent_variables_80.json"
list_best_vars = read_best_variables(json_best_variables)

N_ITERATIONS = 20
BASE_SEED = 42
N_THREADS = 6


def get_models_config_for_seed(seed):
    """Retorna una copia de la configuracion de modelos usando una semilla indicada."""
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


def _set_cfg(class_path):
    CFG.class_path = class_path
    os.makedirs(CFG.class_path, exist_ok=True)
    CFG.individual_train = False
    CFG.cuartiles_train = True
    CFG.include_prod = False


def _train_non_nested(model_name, model_config, seed, best_vars):
    # Ensure worker process has cuartiles flags.
    CFG.individual_train = False
    CFG.cuartiles_train = True
    CFG.include_prod = False
    dir_path = f"{CFG.class_path}{model_name.replace(' ', '_')}/"
    os.makedirs(dir_path, exist_ok=True)
    result = train_test_model(
        df_imputed=df_imputed,
        n_clases=None,
        model_name=model_name,
        model_config=model_config,
        element="Quartiles",
        usar_smote=False,
        mostrar_graficos=True,
        calcular_shap=True,
        dir_path=dir_path,
        CFG=CFG,
        best_variables=best_vars,
        seed=seed,
    )
    return model_name, result


def run_non_nested_iteration(models_config, class_path, seed, best_vars):
    #_set_cfg(class_path)
    CFG.class_path = class_path
    os.makedirs(CFG.class_path, exist_ok=True)
    CFG.individual_train = False
    CFG.cuartiles_train = True
    CFG.include_prod = False
    CFG.path_pkl_results_classification = f"{CFG.class_path}class_models_cuartiles_all_models.pkl"

    jobs = (
        delayed(_train_non_nested)(model_name, model_config, seed, best_vars)
        for model_name, model_config in models_config.items()
    )
    all_results_list = Parallel(n_jobs=N_THREADS, verbose=10)(jobs)

    all_results = {}
    for model_name, result in all_results_list:
        all_results.setdefault(model_name, []).append(result)

    with open(CFG.path_pkl_results_classification, 'wb') as pkl_file:
        pickle.dump(all_results, pkl_file)

    compare_classification_models(all_results, CFG=CFG)
    save_results_general(all_results, CFG.class_path)


def _train_nested(model_name, model_config, seed, best_vars):
    # Ensure worker process has cuartiles flags.
    CFG.individual_train = False
    CFG.cuartiles_train = True
    CFG.include_prod = False
    dir_path = f"{CFG.class_path}{model_name.replace(' ', '_')}/"
    os.makedirs(dir_path, exist_ok=True)
    result = train_test_class_nested(
        df_imputed=df_imputed,
        n_clases=None,
        model_name=model_name,
        model_config=model_config,
        element="Quartiles",
        usar_smote=False,
        mostrar_graficos=True,
        calcular_shap=True,
        dir_path=dir_path,
        best_variables=best_vars,
        CFG=CFG,
        seed=seed,
    )
    return model_name, result


def run_nested_iteration(models_config, class_path, seed, best_vars):
    #_set_cfg(class_path)
    CFG.class_path = class_path
    os.makedirs(CFG.class_path, exist_ok=True)
    CFG.individual_train = False
    CFG.cuartiles_train = True
    CFG.include_prod = False
    CFG.path_pkl_results_classification = f"{CFG.class_path}class_models_cuartiles_all_models.pkl"

    jobs = (
        delayed(_train_nested)(model_name, model_config, seed, best_vars)
        for model_name, model_config in models_config.items()
    )
    all_results_list = Parallel(n_jobs=N_THREADS, verbose=10)(jobs)

    all_results = {}
    for model_name, result in all_results_list:
        all_results.setdefault(model_name, []).append(result)

    with open(CFG.path_pkl_results_classification, 'wb') as pkl_file:
        pickle.dump(all_results, pkl_file)

    compare_classification_models(all_results, CFG=CFG)
    save_results_general(all_results, CFG.class_path)


def run_all_models_iteration(models_config, class_path, seed, best_vars):
    #_set_cfg(class_path)
    CFG.class_path = class_path
    os.makedirs(CFG.class_path, exist_ok=True)
    CFG.individual_train = False
    CFG.cuartiles_train = True
    CFG.include_prod = False
    CFG.path_pkl_results_classification = f"{CFG.class_path}class_models_cuartiles_all_models.pkl"

    all_results = {}
    for model_name, model_config in models_config.items():
        dir_path = f"{CFG.class_path}Quartiles/"
        os.makedirs(dir_path, exist_ok=True)
        result = train_test_model_all_predictions(
            df_imputed=df_imputed,
            n_clases=None,
            model_name=model_name,
            model_config=model_config,
            element="Quartiles",
            usar_smote=False,
            mostrar_graficos=True,
            calcular_shap=True,
            dir_path=dir_path,
            best_variables=best_vars,
            CFG=CFG,
            seed=seed,
        )
        all_results.setdefault(model_name, []).append(result)

    with open(CFG.path_pkl_results_classification, 'wb') as pkl_file:
        pickle.dump(all_results, pkl_file)

    compare_classification_models(all_results, CFG=CFG)
    save_results_general(all_results, CFG.class_path)


def run_pca_iteration(models_config, class_path, seed, best_vars):
    #_set_cfg(class_path)
    CFG.class_path = class_path
    os.makedirs(CFG.class_path, exist_ok=True)
    CFG.individual_train = False
    CFG.cuartiles_train = True
    CFG.include_prod = False
    CFG.path_pkl_results_classification = f"{CFG.class_path}class_models_cuartiles.pkl"

    jobs = (
        delayed(train_test_model_pca)(
            df_imputed=df_imputed,
            n_clases=None,
            model_name=model_name,
            model_config=model_config,
            element="Quartiles",
            usar_smote=False,
            mostrar_graficos=True,
            calcular_shap=False,
            dir_path=f"{CFG.class_path}{model_name.replace(' ', '_')}/",
            n_components=2,
            CFG=CFG,
            best_variables=best_vars,
            seed=seed,
        )
        for model_name, model_config in models_config.items()
    )
    all_results_list = Parallel(n_jobs=N_THREADS, verbose=10)(jobs)

    all_results = {}
    for model_name, result in zip(models_config.keys(), all_results_list):
        all_results.setdefault(model_name, []).append(result)

    with open(CFG.path_pkl_results_classification, 'wb') as pkl_file:
        pickle.dump(all_results, pkl_file)

    save_results_general(all_results, CFG.class_path)


def run_iteration(iteration_idx, seed, best_vars):
    print(f"\nIteracion {iteration_idx}/{N_ITERATIONS} | seed={seed}")
    models_config_seed = get_models_config_for_seed(seed)

    run_non_nested_iteration(
        models_config=models_config_seed,
        class_path=(
            f"{CFG.Root}/Resultados/classification_cuartiles_less_vars/"
            f"iter_{iteration_idx:02d}_seed_{seed}/"
        ),
        seed=seed,
        best_vars=best_vars,
    )

    run_nested_iteration(
        models_config=models_config_seed,
        class_path=(
            f"{CFG.Root}/Resultados/classification_cuartiles_less_vars_nested/"
            f"iter_{iteration_idx:02d}_seed_{seed}/"
        ),
        seed=seed,
        best_vars=best_vars,
    )
'''
    run_all_models_iteration(
        models_config=models_config_seed,
        class_path=(
            f"{CFG.Root}/Resultados/classification_cuartiles_less_vars_all_models/"
            f"iter_{iteration_idx:02d}_seed_{seed}/"
        ),
        seed=seed,
        best_vars=best_vars,
    )

    run_pca_iteration(
        models_config=models_config_seed,
        class_path=(
            f"{CFG.Root}/Resultados/classification_cuartiles_less_vars_pca/"
            f"iter_{iteration_idx:02d}_seed_{seed}/"
        ),
        seed=seed,
        best_vars=best_vars,
    )
'''

for iteration_idx in range(1, N_ITERATIONS + 1):
    seed = BASE_SEED + iteration_idx - 1
    run_iteration(iteration_idx, seed, list_best_vars)
