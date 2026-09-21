import os
import json
import pickle
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import CFG, MODELS_CONFIG, setup_logger
from new_utils import impute_data, extract_frequent_values_from_csv
from xai_utils import compute_shap_importance
from cuartiles_utils import (
    train_test_cuartiles, 
    run_permutation_cuartiles
)

# 0. CONFIGURACIÓN Y LOGGER
N_ITERATIONS = 20
BASE_SEED = 42
SHAP_ITERATIONS = 10
TARGET_METRIC = 'f1_test_macro'

output_dir = os.path.join(CFG.Root, "Resultados", "pipeline_cuartiles")
xai_dir = os.path.join(output_dir, "xai_outputs")
os.makedirs(xai_dir, exist_ok=True)

checkpoint_file = os.path.join(output_dir, "checkpoint_cuartiles.json")
progress_file = os.path.join(output_dir, "historico_progress.pkl")

if os.path.exists(checkpoint_file):
    with open(checkpoint_file, 'r', encoding='utf-8') as file:
        checkpoint = json.load(file)
else:
    checkpoint = {'completadas': []}

completed_iterations = set(checkpoint.get('completadas', []))

if os.path.exists(progress_file):
    with open(progress_file, 'rb') as file:
        historico = pickle.load(file)
else:
    historico = {}

logger = setup_logger("Cuartiles_Pipeline", log_file=os.path.join(output_dir, "pipeline.log"))

# 1. CARGA DE DATOS
logger.info(f"Cargando dataset desde: {CFG.data_path_clean}")
data_clean = pd.read_csv(CFG.data_path_clean)

# 2. BUCLE DE 20 ITERACIONES (SPLIT, IMPUTACIÓN, ENTRENAMIENTO)
logger.info(f"Iniciando {N_ITERATIONS} iteraciones de entrenamiento por cuartiles...")

for idx in range(N_ITERATIONS):
    seed = BASE_SEED + idx
    iteration_key = f"iteration_{idx}_seed_{seed}"

    if iteration_key in completed_iterations and idx in historico:
        logger.info(f"Saltando iteración {idx + 1}/{N_ITERATIONS}: ya está registrada")
        continue

    logger.info(f"Iteración {idx + 1}/{N_ITERATIONS} (Semilla: {seed})")
    
    # Split 70/30 estratificado por tratamiento
    data_clean['Etiqueta_NPK'] = data_clean['Tratamiento'].str.extract(r'(N\dP\dK\d)')
    train_data, test_data = train_test_split(
        data_clean, test_size=0.3, random_state=seed, stratify=data_clean['Etiqueta_NPK']
    )
    
    train_imp, test_imp = impute_data(train_data, test_data, seed=seed)
    historico[idx] = {}
    
    for model_name, model_cfg in MODELS_CONFIG.items():
        logger.info(f"Modelo '{model_name}' - Iteración {idx + 1}...")
        res = train_test_cuartiles(
            train_data=train_imp,
            test_data=test_imp,
            model_name=model_name,
            model_config=model_cfg,
            treat_quantiles_path=CFG.treat_quantiles_path,
            productivity_vars=CFG.productivity_vars,
            include_prod=False,
            seed=seed
        )
        historico[idx][model_name] = res

    completed_iterations.add(iteration_key)
    with open(progress_file, 'wb') as file:
        pickle.dump(historico, file)
    with open(checkpoint_file, 'w', encoding='utf-8') as file:
        json.dump({'completadas': sorted(completed_iterations)}, file, indent=4)
    logger.info(f"Checkpoint guardado: {iteration_key}")

# 3. SELECCIÓN DE LA MEJOR ITERACIÓN
logger.info("Evaluando la iteración con mejor rendimiento promedio...")
mejor_iter = -1
mejor_score = -1.0

for idx in range(N_ITERATIONS):
    scores = [res[TARGET_METRIC] for res in historico[idx].values()]
    promedio = float(np.mean(scores))
    if promedio > mejor_score:
        mejor_score = promedio
        mejor_iter = idx

logger.info(f"Ganadora: Iteración {mejor_iter + 1} ({TARGET_METRIC} = {mejor_score:.4f})")
best_results = historico[mejor_iter]

# 4. INTERPRETABILIDAD SHAP BINARIA (10 ITERACIONES)
logger.info("Calculando SHAP para 2 clases (10 iteraciones)...")
shap_dir = os.path.join(xai_dir, "shap")
os.makedirs(shap_dir, exist_ok=True)
common_vars_alg = {}

for model_name, model_cfg in MODELS_CONFIG.items():
    res = best_results[model_name]
    pipeline = res['grid_search'].best_estimator_
    scaler = pipeline.named_steps['scaler']
    clf = pipeline.named_steps['clf']
    
    X_test = res['X_test']
    features = res['feature_names']
    X_scaled = pd.DataFrame(scaler.transform(X_test), columns=features)
    
    alg_iters = {}
    for i in range(SHAP_ITERATIONS):
        plot_p = os.path.join(shap_dir, f"shap_{model_name}.png") if i == 0 else None
        imp = compute_shap_importance(clf,
                                      X_scaled,
                                      features,
                                      model_cfg['model_type'],
                                      i,
                                      BASE_SEED,
                                      output_plot_path=plot_p,
                                      logger=logger)
        
        # Filtro 80%
        df_imp = pd.DataFrame({'f': features, 'imp': imp}).sort_values('imp', ascending=False)
        norm = df_imp['imp'] / df_imp['imp'].sum()
        top = df_imp[norm.cumsum() <= 0.80]['f'].tolist()
        alg_iters[f"Iteration_{i+1}"] = top if top else df_imp['f'].iloc[:1].tolist()
        
    csv_alg = os.path.join(shap_dir, f"vars_{model_name}.csv")
    pd.DataFrame(dict([(k, pd.Series(v)) for k, v in alg_iters.items()])).to_csv(csv_alg, index=False)
    common_vars_alg[model_name] = extract_frequent_values_from_csv(csv_alg, threshold_percentage=80)

# Consenso SHAP entre modelos (>= 80% de modelos)
csv_common = os.path.join(shap_dir, "common_vars_cuartiles_80.csv")
pd.DataFrame(dict([(k, pd.Series(v)) for k, v in common_vars_alg.items()])).to_csv(csv_common, index=False)
final_shap_vars = extract_frequent_values_from_csv(csv_common, threshold_percentage=80)
final_shap_vars_100 = extract_frequent_values_from_csv(csv_common, threshold_percentage=100)
logger.info(f"Variables SHAP >= 80% ({len(final_shap_vars)}): {final_shap_vars}")
logger.info(f"Variables SHAP 100% ({len(final_shap_vars_100)}): {final_shap_vars_100}")

# 5. PERMUTATION IMPORTANCE
logger.info("Calculando Permutation Importance...")
perm_vars = run_permutation_cuartiles(best_results, MODELS_CONFIG, os.path.join(xai_dir, "permutation"), BASE_SEED)
perm_vars_80 = perm_vars[80]
perm_vars_100 = perm_vars[100]
logger.info(f"Variables Permutation >= 80% ({len(perm_vars_80)}): {perm_vars_80}")
logger.info(f"Variables Permutation 100% ({len(perm_vars_100)}): {perm_vars_100}")

# 6. INTERSECCIONES FINALES POR NIVEL DE CONSENSO
cuartiles_best_vars_80 = sorted(list(set(final_shap_vars).intersection(set(perm_vars_80))))
cuartiles_best_vars_100 = sorted(list(set(final_shap_vars_100).intersection(set(perm_vars_100))))
logger.info(f"Variables SHAP/Permutation al 80% ({len(cuartiles_best_vars_80)}): {cuartiles_best_vars_80}")
logger.info(f"Variables SHAP/Permutation al 100% ({len(cuartiles_best_vars_100)}): {cuartiles_best_vars_100}")

pd.DataFrame({'Cuartiles_Best_Vars': cuartiles_best_vars_80}).to_csv(
    os.path.join(xai_dir, "cuartiles_best_vars_80.csv"), index=False
)
pd.DataFrame({'Cuartiles_Best_Vars': cuartiles_best_vars_100}).to_csv(
    os.path.join(xai_dir, "cuartiles_best_vars_100.csv"), index=False
)
logger.info("Pipeline de cuartiles finalizado con éxito.")