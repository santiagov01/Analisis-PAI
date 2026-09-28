import pickle
import pandas as pd
import numpy as np
import os
from config import CFG, MODELS_CONFIG, setup_logger

from new_utils import (
    impute_data,
    train_test_NPK,
    split_data_NPK,
    save_results_iterations
)
from xai_utils import (
    run_shap_pipeline,
    run_permutation_pipeline,
    extract_and_save_robust_features
)

def read_pkl(file_path):
    """
    Reads a pickle file and returns its content.

    Args:
        file_path (str): Path to the pickle file.
    """
    with open(file_path, 'rb') as file:
        return pickle.load(file)

# configurar logger

CFG.xai_output_dir = f'{CFG.Root}/Resultados/all_xai_outputs_npk' 
log_file = os.path.join(CFG.xai_output_dir, "xai_all_iters.log")
logger = setup_logger(name="XAI_all_iters", log_file=log_file)



path_pkl_npk = "/export2/svargash/PAI-Quindio/Code/Analisis-PAI/Resultados/pipeline_imputation_NPK/historico_resultados.pkl"

#===========
# recorrer todas las iteraciones guardadas para npk

# leer archivo pkl de npk
resultados_npk = read_pkl(path_pkl_npk)
results = {}
# Organizar los resultados como results[iteracion][modelo].
#calcular shap y permutacion para cada modelo y sus iteraciones
for modelo, values in resultados_npk.items():
    for iteracion, resultados in values.items():
        print(f"Iteración: {iteracion}, Modelo: {modelo}")
        results.setdefault(iteracion, {})[modelo] = resultados
del resultados_npk  # Liberar memoria
for iteracion, modelos in results.items():
    # Ejecutar SHAP
    #configurar ruta por iteracion:
    CFG.xai_output_dir = f'{CFG.Root}/Resultados/all_xai_outputs_npk/iteracion_{iteracion}'
    shap_results = run_shap_pipeline(
        best_results=results[iteracion],
        elements=CFG.elements_list,
        models_config=MODELS_CONFIG,
        output_dir=CFG.xai_output_dir,
        n_iterations=CFG.shap_iterations,
        base_seed=CFG.base_seed,
        logger=logger
    )

    perm_results = run_permutation_pipeline(
    best_results=results[iteracion],
    elements=CFG.elements_list,
    models_config=MODELS_CONFIG,
    output_dir=CFG.xai_output_dir,
    base_seed=CFG.base_seed,
    logger=logger
    )
    extract_and_save_robust_features(
        shap_results=shap_results,
        perm_results=perm_results,
        elements=CFG.elements_list,
        output_dir=CFG.xai_output_dir,
        logger=logger
    )
del results  # Liberar memoria