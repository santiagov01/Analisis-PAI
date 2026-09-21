# ==============================================================================
# MÓDULO DE INTERPRETABILIDAD (XAI): SHAP Y PERMUTATION IMPORTANCE
# ==============================================================================

import os
import json
import numpy as np
import pandas as pd
import seaborn as sns

# Configuración no interactiva de Matplotlib (headless)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import shap
from eli5.sklearn import PermutationImportance

def clean_feature_names(feature_names):
    """Convierte nombres incompatibles con XGBoost en nombres válidos."""
    cleaned_names = []

    for name in feature_names:
        cleaned_name = (
            str(name)
            .replace("[", "(")
            .replace("]", ")")
            .replace("<", "_lt_")
        )
        cleaned_names.append(cleaned_name)

    return cleaned_names

def compute_shap_importance(clf, X_scaled_df, feature_names, model_type='tree', iteration=0, base_seed=42, output_plot_path=None, logger=None):
    """Calcula importancias SHAP para un modelo específico.
    
    - Modelos de árbol: TreeExplainer sobre todo el conjunto de prueba.
    - Modelos kernel (SVM, KNN, MLP): KernelExplainer sobre 100 muestras aleatorias.
    """
    original_feature_names = list(feature_names)
    safe_feature_names = clean_feature_names(original_feature_names)

    X_scaled_safe = X_scaled_df.copy()
    X_scaled_safe.columns = safe_feature_names

    X_used = X_scaled_safe
    if model_type == 'tree':
        try:
            # Para modelos de árbol, usar TreeExplainer directamente
            if hasattr(clf, 'get_booster'):
                logger.info(f"Usando TreeExplainer con model_output='raw' para XGBoost en iteración {iteration + 1}")

                booster = clf.get_booster()
                # 1. Resolver discrepancia de nombres de columnas
                # Asignar los feature_names reales al booster para que coincidan con X_scaled_df
                booster.feature_names = safe_feature_names
                
                # Usar TreeExplainer directamente sobre el booster corregido
                explainer = shap.TreeExplainer(booster, model_output='raw')
                shap_vals = explainer.shap_values(X_scaled_safe)
            else:
                # Para Random Forest u otros modelos de árbol
                explainer = shap.TreeExplainer(clf)
                shap_vals = explainer.shap_values(X_scaled_safe)
        except Exception:
            # En caso de error, usar KernelExplainer como fallback
            logger.warning(f"TreeExplainer falló para el modelo {clf}. Usando KernelExplainer como fallback.")
            background = shap.sample(X_scaled_df, min(100, len(X_scaled_df)), random_state=base_seed + iteration)
            explainer = shap.KernelExplainer(clf.predict_proba, background)
            shap_vals = explainer.shap_values(background, nsamples=100)
            X_used = background
    else:
        logger.info(f"Usando KernelExplainer para el modelo {clf} en iteración {iteration + 1}")
        background = shap.sample(X_scaled_df, min(100, len(X_scaled_df)), random_state=base_seed + iteration)
        explainer = shap.KernelExplainer(clf.predict_proba, background)
        shap_vals = explainer.shap_values(background, nsamples=100)
        X_used = background

    # Unificación de dimensiones: Promedio absoluto de SHAP
    if isinstance(shap_vals, list):
        # Para modelos multiclase, shap_vals es una lista de arrays; se promedia sobre clases y muestras
        shap_importance = np.mean([np.mean(np.abs(sv), axis=0) for sv in shap_vals], axis=0)
    elif hasattr(shap_vals, 'ndim') and shap_vals.ndim == 3:
        # Para modelos multiclase con shap_vals de 3D, se promedia sobre clases y muestras
        shap_importance = np.mean(np.abs(shap_vals), axis=(0, 2))
    else:
        # Para modelos binarios o regresión, shap_vals es un array 2D; se promedia sobre muestras
        shap_importance = np.mean(np.abs(shap_vals), axis=0)

    # Generación y guardado del gráfico (sin mostrar en pantalla)
    if output_plot_path:
        fig = plt.figure(figsize=(10, 5))
        shap.summary_plot(
                shap_vals,
                X_used,
                feature_names=original_feature_names,
                plot_type="bar",
                show=False
                    )
        plt.title(f"SHAP Feature Importance (Iteración {iteration + 1})", fontsize=12, pad=15)
        plt.tight_layout()
        plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    return shap_importance


def run_shap_pipeline(best_results, elements, models_config, output_dir, n_iterations=10, base_seed=42, logger=None):
    """Ejecuta el flujo completo de SHAP:
    
    10 iteraciones variando la porcion de datos de test
    Filtrado 80% acumulado
    Consenso iteraciones
    Consenso entre modelos.
    """
    shap_results_by_element = {}
    shap_results_by_element_100 = {}

    for element in elements:
        if logger:
            logger.info(f"Iniciando análisis SHAP para elemento: {element}")
            
        element_dir = os.path.join(output_dir, "shap", element)
        os.makedirs(element_dir, exist_ok=True)
        
        # Inicialización de diccionario para almacenar las variables seleccionadas por cada algoritmo y cada iteración
        features_per_algorithm = {model_name: {} for model_name in models_config.keys()}

        for model_name, model_config in models_config.items():
            if logger:
                logger.info(f"Calculando SHAP para {element} con algoritmo {model_name} ({model_config['model_type']})")
            # Recuperación de resultados del modelo entrenado    
            res = best_results[element][model_name]
            pipeline = res['grid_search'].best_estimator_
            scaler = pipeline.named_steps['scaler']
            clf = pipeline.named_steps['clf']
            # Recuperación de datos de prueba y nombres de características
            X_test = res['X_test']
            feature_names = res['feature_names']
            X_scaled_df = pd.DataFrame(scaler.transform(X_test), columns=feature_names)

            for i in range(n_iterations):
                plot_path = os.path.join(element_dir, f"shap_bar_{model_name}_iter_{i+1}.png") if i == 0 else None
                
                shap_imp = compute_shap_importance(
                    clf=clf,
                    X_scaled_df=X_scaled_df,
                    feature_names=feature_names,
                    model_type=model_config['model_type'],
                    iteration=i,
                    base_seed=base_seed,
                    output_plot_path=plot_path,
                    logger=logger
                )

                # Acumulación hasta el 80%
                df_imp = pd.DataFrame({'feature': feature_names, 'importance': shap_imp})
                df_imp = df_imp.sort_values(by='importance', ascending=False).reset_index(drop=True)
                total_sum = df_imp['importance'].sum()
                df_imp['normalized'] = df_imp['importance'] / total_sum if total_sum > 0 else 0

                top_features = []
                acc = 0.0
                for _, row in df_imp.iterrows():
                    top_features.append(row['feature'])
                    acc += row['normalized']
                    if acc >= 0.80:
                        break

                features_per_algorithm[model_name][f"Iteration_{i+1}"] = top_features

            # Guardar iteraciones del algoritmo
            df_alg_iters = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in features_per_algorithm[model_name].items()]))
            df_alg_iters.to_csv(os.path.join(element_dir, f"vars_{model_name}.csv"), index=False)

        # Consenso por algoritmo (aparece en >= 80% de las ITERACIONES)
        common_vars_per_alg = {}
        for model_name in models_config.keys():
            csv_path = os.path.join(element_dir, f"vars_{model_name}.csv")
            common_vars = _extract_frequent_features(csv_path, threshold_pct=80)
            common_vars_per_alg[model_name] = common_vars

        df_common_alg = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in common_vars_per_alg.items()]))
        common_csv_path = os.path.join(element_dir, f"common_vars_{element}_80.csv")
        df_common_alg.to_csv(common_csv_path, index=False)

        # Consenso final entre algoritmos (aparece en >= 80% de los ALGORITMOS)
        final_element_vars = _extract_frequent_features(common_csv_path, threshold_pct=80)
        shap_results_by_element[element] = final_element_vars

        # Consenso final entre algoritmos (aparece en >= 100% de los ALGORITMOS)
        final_element_vars_100 = _extract_frequent_features(common_csv_path, threshold_pct=100)
        shap_results_by_element_100[element] = final_element_vars_100
        
        if logger:
            logger.info(f"Variables SHAP consensuadas para {element}: {len(final_element_vars)} variables identificadas")

    # Guardado global 80% y 100%
    df_global = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in shap_results_by_element.items()]))
    df_global.to_csv(os.path.join(output_dir, "shap_common_vars_all_elements_80.csv"), index=False)
    with open(os.path.join(output_dir, "shap_results_summary.json"), 'w') as f:
        json.dump(shap_results_by_element, f, indent=4)

    df_global_100 = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in shap_results_by_element_100.items()]))
    df_global_100.to_csv(os.path.join(output_dir, "shap_common_vars_all_elements_100.csv"), index=False)
    with open(os.path.join(output_dir, "shap_results_summary_100.json"), 'w') as f:
        json.dump(shap_results_by_element_100, f, indent=4)

    return shap_results_by_element


def run_permutation_pipeline(best_results, elements, models_config, output_dir, base_seed=42, n_iter=15, logger=None):
    """Ejecuta el flujo completo de Permutation Importance:
    
    Permutación sobre Pipeline -> filtrado 80% -> análisis de frecuencias y ranking de consenso.
    """
    perm_results_by_element = {}
    perm_dir = os.path.join(output_dir, "permutation_importance")
    os.makedirs(perm_dir, exist_ok=True)

    for element in elements:
        if logger:
            logger.info(f"Iniciando Permutation Importance para elemento: {element}")
        # Creación de DataFrame para almacenar resultados de permutación
        first_model = list(models_config.keys())[0]
        feature_names = best_results[element][first_model]['feature_names']

        df_perm = pd.DataFrame({'Feature': feature_names})
        dict_norm_vals = {}

        for model_name in models_config.keys():
            # Recuperación de resultados del modelo entrenado
            res = best_results[element][model_name]
            pipeline = res['grid_search'].best_estimator_
            X_test = res['X_test']
            y_test = res['y_test']
            # Cálculo de Permutation Importance
            perm = PermutationImportance(pipeline, random_state=base_seed, n_iter=n_iter, cv="prefit").fit(X_test, y_test)
            # Normalización de importancias
            importances = perm.feature_importances_ #
            # Normalización de importancias (evitando negativos)
            importances[importances < 0] = 0.0
            sum_imp = np.sum(importances)
            norm_imp = importances / sum_imp if sum_imp > 0 else importances

            dict_norm_vals[model_name] = norm_imp
            df_perm[model_name] = norm_imp

        df_perm.to_csv(os.path.join(perm_dir, f"permutation_importance_{element}.csv"), index=False)

        # Selección de top 80% acumulado
        best_feats_by_alg = {}
        for model_name in models_config.keys():
            df_sorted = pd.DataFrame({
                'Variable': feature_names,
                'Importance': dict_norm_vals[model_name]
            }).sort_values(by='Importance', ascending=False).reset_index(drop=True)

            df_sorted['Cumulative'] = df_sorted['Importance'].cumsum()
            top_feats = df_sorted[df_sorted['Cumulative'] <= 0.80]['Variable'].tolist()
            if not top_feats:
                top_feats = df_sorted['Variable'].iloc[:1].tolist()
            best_feats_by_alg[model_name] = top_feats
        # Guardado de top 80% por algoritmo
        top_perm_csv = os.path.join(perm_dir, f"best_80_percent_features_{element}.csv")
        df_top = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in best_feats_by_alg.items()]))
        df_top.to_csv(top_perm_csv, index=False)

        # Análisis de frecuencia y consenso
        common_vars = _analyze_frequent_permutation_vars(
            csv_path=top_perm_csv,
            element=element,
            output_dir=perm_dir
        )
        perm_results_by_element[element] = common_vars
        
        if logger:
            logger.info(f"Variables Permutation Importance para {element}: {len(common_vars)} variables identificadas")

    # Guardado global
    df_global = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in perm_results_by_element.items()]))
    df_global.to_csv(os.path.join(perm_dir, "most_frequent_variables_TOTAL_80.csv"), index=False)
    with open(os.path.join(perm_dir, "most_frequent_variables_80.json"), 'w') as f:
        json.dump(perm_results_by_element, f, indent=4)

    return perm_results_by_element


def extract_and_save_robust_features(shap_results, perm_results, elements, output_dir, logger=None):
    """Calcula y guarda la intersección entre las variables de SHAP y Permutation Importance."""
    robust_features = {}

    for element in elements:
        shap_set = set(shap_results.get(element, []))
        perm_set = set(perm_results.get(element, []))
        intersection = sorted(list(shap_set.intersection(perm_set)))
        robust_features[element] = intersection
        
        if logger:
            logger.info(f"Variables validadas por ambos métodos para {element} ({len(intersection)}): {intersection}")

    df_robust = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in robust_features.items()]))
    df_robust.to_csv(os.path.join(output_dir, "final_robust_features_SHAP_and_PERMUTATION.csv"), index=False)
    with open(os.path.join(output_dir, "final_robust_features.json"), 'w') as f:
        json.dump(robust_features, f, indent=4)

    return robust_features


# ==============================================================================
# FUNCIONES AUXILIARES INTERNAS DE XAI
# ==============================================================================
def _extract_frequent_features(csv_path, threshold_pct=80):
    """Extrae las variables que superan un umbral de presencia en columnas y las ordena por ranking agregado."""
    df = pd.read_csv(csv_path)
    column_rankings, column_sets = [], []
    
    for col in df.columns:
        rank_dict = {}
        for pos, val in enumerate(df[col]):
            if pd.notna(val) and str(val).strip() != '':
                rank_dict[val] = pos
        column_rankings.append(rank_dict)
        column_sets.append(set(rank_dict.keys()))

    if not column_sets:
        return []

    min_appearances = int(np.ceil(len(column_sets) * threshold_pct / 100))
    all_values = set.union(*column_sets)
    frequent_values = [v for v in all_values if sum(1 for c in column_sets if v in c) >= min_appearances]

    scores = {v: sum(r[v] for r in column_rankings if v in r) for v in frequent_values}
    return sorted(frequent_values, key=lambda v: scores[v])


def _analyze_frequent_permutation_vars(csv_path, element, output_dir):
    """Analiza frecuencias en permutation importance y guarda el gráfico sin mostrarlo."""
    df = pd.read_csv(csv_path)
    freq_dict, pos_dict = {}, {}

    for col in df.columns:
        for pos, var in enumerate(df[col].dropna(), start=1):
            freq_dict[var] = freq_dict.get(var, 0) + 1
            pos_dict.setdefault(var, []).append(pos)

    if not freq_dict:
        return []

    summary = pd.DataFrame([
        {'Variable': v, 'Frequency': freq_dict[v], 'Position_Sum': sum(pos_dict[v])}
        for v in freq_dict
    ]).sort_values(by=['Frequency', 'Position_Sum'], ascending=[False, True]).reset_index(drop=True)

    # Gráfico de frecuencias guardado directamente
    fig = plt.figure(figsize=(9, 5))
    plot_df = summary.copy()
    plot_df['Variable'] = plot_df['Variable'].str.replace('_', ' ')
    sns.barplot(data=plot_df, x='Frequency', y='Variable', color='steelblue')
    plt.title(f'Frecuencia de Variables Top 80% - Modelo {element}', fontsize=12, fontweight='bold')
    plt.xlabel('Frecuencia entre algoritmos', fontsize=10)
    plt.ylabel('Variable', fontsize=10)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'top_vars_freq_80_{element}.png'), dpi=300)
    plt.close(fig)

    all_models_count = len(df.columns)
    return summary[summary['Frequency'] == all_models_count]['Variable'].tolist()