import pickle
import pandas as pd
import numpy as np
import os
from config import CFG, MODELS_CONFIG, setup_logger

from new_utils import (
    impute_data,
    train_test_NPK,
    split_data_NPK,
    save_results_iterations,
    extract_frequent_values_from_csv
)
from xai_utils import (
    compute_shap_importance,
    run_shap_pipeline,
    run_permutation_pipeline,
    extract_and_save_robust_features
)
from cuartiles_utils import run_permutation_cuartiles

def read_pkl(file_path):
    """
    Reads a pickle file and returns its content.

    Args:
        file_path (str): Path to the pickle file.
    """
    with open(file_path, 'rb') as file:
        return pickle.load(file)


def run_cuartiles_xai_iteration(
    iteration_results,
    iteration_name,
    output_dir,
    model_configs,
    base_seed=42,
    shap_iterations=10,
    logger=None,
):
    """Run the XAI pipeline for one iteration from historico."""
    if logger:
        logger.info(f"Iniciando XAI para la iteración {iteration_name}")

    shap_dir = os.path.join(output_dir, "shap")
    os.makedirs(shap_dir, exist_ok=True)
    common_vars_alg = {}

    for model_name, model_cfg in model_configs.items():
        res = iteration_results[model_name]
        pipeline = res['grid_search'].best_estimator_
        scaler = pipeline.named_steps['scaler']
        clf = pipeline.named_steps['clf']

        X_test = res['X_test']
        features = res['feature_names']
        X_scaled = pd.DataFrame(scaler.transform(X_test), columns=features)

        alg_iters = {}
        for shap_iteration in range(shap_iterations):
            plot_path = (
                os.path.join(shap_dir, f"shap_{model_name}.png")
                if shap_iteration == 0 else None
            )
            importance = compute_shap_importance(
                clf,
                X_scaled,
                features,
                model_cfg['model_type'],
                shap_iteration,
                base_seed,
                output_plot_path=plot_path,
                logger=logger,
            )

            df_importance = pd.DataFrame({
                'f': features,
                'imp': importance,
            }).sort_values('imp', ascending=False)
            normalized = df_importance['imp'] / df_importance['imp'].sum()
            top_features = df_importance[
                normalized.cumsum() <= 0.80
            ]['f'].tolist()
            alg_iters[f"Iteration_{shap_iteration + 1}"] = (
                top_features or df_importance['f'].iloc[:1].tolist()
            )

        csv_alg = os.path.join(shap_dir, f"vars_{model_name}.csv")
        pd.DataFrame(dict([
            (key, pd.Series(value)) for key, value in alg_iters.items()
        ])).to_csv(csv_alg, index=False)
        common_vars_alg[model_name] = extract_frequent_values_from_csv(
            csv_alg, threshold_percentage=80
        )

    csv_common = os.path.join(shap_dir, "common_vars_cuartiles_80.csv")
    pd.DataFrame(dict([
        (key, pd.Series(value)) for key, value in common_vars_alg.items()
    ])).to_csv(csv_common, index=False)
    final_shap_vars = extract_frequent_values_from_csv(
        csv_common, threshold_percentage=80
    )
    final_shap_vars_100 = extract_frequent_values_from_csv(
        csv_common, threshold_percentage=100
    )
    # PERMUTATION IMPORTANCE
    if logger:
        logger.info("Calculando Permutation Importance...")
    permutation_dir = os.path.join(output_dir, "permutation")
    perm_vars = run_permutation_cuartiles(
        iteration_results, model_configs, permutation_dir, base_seed
    )
    perm_vars_80 = perm_vars[80]
    perm_vars_100 = perm_vars[100]

    best_vars_80 = sorted(set(final_shap_vars).intersection(perm_vars_80))
    best_vars_100 = sorted(set(final_shap_vars_100).intersection(perm_vars_100))

    pd.DataFrame({'Cuartiles_Best_Vars': best_vars_80}).to_csv(
        os.path.join(output_dir, "cuartiles_best_vars_80.csv"), index=False
    )
    pd.DataFrame({'Cuartiles_Best_Vars': best_vars_100}).to_csv(
        os.path.join(output_dir, "cuartiles_best_vars_100.csv"), index=False
    )

    if logger:
        logger.info(
            f"Iteración {iteration_name}: "
            f"SHAP 80%={len(final_shap_vars)}, "
            f"Permutation 80%={len(perm_vars_80)}, "
            f"Intersección 80%={len(best_vars_80)}"
        )

    return {
        'shap_vars_80': final_shap_vars,
        'shap_vars_100': final_shap_vars_100,
        'permutation_vars_80': perm_vars_80,
        'permutation_vars_100': perm_vars_100,
        'best_vars_80': best_vars_80,
        'best_vars_100': best_vars_100,
    }

# configurar logger

CFG.xai_output_dir = f'{CFG.Root}/Resultados/all_xai_outputs_cuartiles' 
log_file = os.path.join(CFG.xai_output_dir, "xai_all_iters.log")
logger = setup_logger(name="XAI_all_iters", log_file=log_file)



path_pkl_cuartiles = "/export2/svargash/PAI-Quindio/Code/Analisis-PAI/Resultados/pipeline_cuartiles/historico_progress.pkl"

#===========
# recorrer todas las iteraciones guardadas para cuartiles

# leer archivo pkl de npk
resultados_cuartiles = read_pkl(path_pkl_cuartiles)
xai_results = {}

for iteration_name, iteration_results in resultados_cuartiles.items():
    iteration_output_dir = os.path.join(
        CFG.xai_output_dir, f"iteracion_{iteration_name}"
    )
    xai_results[iteration_name] = run_cuartiles_xai_iteration(
        iteration_results=iteration_results,
        iteration_name=iteration_name,
        output_dir=iteration_output_dir,
        model_configs=MODELS_CONFIG,
        base_seed=42,
        shap_iterations=10,
        logger=logger,
    )
    if logger:
        logger.info(f"Finalizada la iteración {iteration_name}")
