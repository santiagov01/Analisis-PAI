from utils import *

import argparse
from pathlib import Path

# =========== LOAD DATA =================

df_imputed = pd.read_csv(CFG.path_df_imputed_corrected)
df_imputed.columns = clean_feature_names(df_imputed.columns)

def parse_args():
    parser = argparse.ArgumentParser(description="Extrae XAI para la mejor iteracion")
    parser.add_argument(
        "--model",
        choices=["cuartiles", "npk"],
        default="cuartiles",
        help="Tipo de modelo: cuartiles o npk",
    )
    parser.add_argument(
        "--best_iter",
        type=int,
        default=5,
        help="Iteracion con mejor performance",
    )
    return parser.parse_args()


def build_model_path(best_iter, model_type):
    best_iter_str = str(best_iter).zfill(2)
    seed = best_iter + 41
    if model_type == "cuartiles":
        return (
            f"{CFG.Root}/Resultados/classification_cuartiles_exclude_prod/"
            f"iter_{best_iter_str}_seed_{seed}/class_models_cuartiles_all_models.pkl"
        )
    return (
        f"{CFG.Root}/Resultados/classification_exclude_prod/"
        f"iter_{best_iter_str}_seed_{seed}/class_results_individual_elements.pkl"
    )


# Cargar modelo de la mejor iteración
args = parse_args()
BEST_ITER = args.best_iter
PATH_MODEL = build_model_path(BEST_ITER, args.model)
if PATH_MODEL:
    with open(PATH_MODEL, 'rb') as pkl_file:
        all_results = pickle.load(pkl_file)
    print(f"Archivo cargado: {PATH_MODEL}")
else:
    print("No se seleccionó ningún archivo")

#obtener el path del directorio del modelo
model_dir = os.path.dirname(PATH_MODEL)
print(f"Directorio del modelo: {model_dir}")

#model_name = os.path.basename(model_dir)
model_name = Path(PATH_MODEL).resolve().parents[1].name

print(f"Nombre del modelo: {model_name}")

if args.model == "cuartiles":
    CFG.cuartiles_train = True
    CFG.individual_train = False
else:
    CFG.cuartiles_train = False
    CFG.individual_train = True

print(f"Entrenamiento por cuartiles: {CFG.cuartiles_train}")
print(f"Entrenamiento individual: {CFG.individual_train}")

def plot_shap_importance(model, X_test, feature_names, model_type='tree', n_clases=2, title="SHAP Feature Importance", path=None, iteration = 0):
    """Genera gráficos de importancia SHAP.

    Args:
        model: Modelo entrenado.
        X_test (DataFrame): Datos de prueba.
        feature_names (list): Nombres de las características.
        model_type (str): 'tree' para TreeExplainer, 'kernel' para KernelExplainer.
                         'kernel' se utiliza para modelos no basados en árboles como SVM, KNN, MLP.
        n_clases (int): Número de clases en el modelo.
        title (str): Título del gráfico.
        path (str): Ruta para guardar el gráfico (sin extensión).
    Returns:
        tuple: (shap_values, explainer, shap_fig, X_df_used)

    """
    X_df = pd.DataFrame(X_test, columns=feature_names)

    if model_type == 'tree':
        try:
            # Try with model_output parameter for better compatibility
            if hasattr(model, 'get_booster'):  # XGBoost specific
                explainer = shap.TreeExplainer(model, model_output='raw')
            else:
                explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_df)
        except (ValueError, AttributeError) as e:
            if "could not convert string to float" in str(e) or "base_score" in str(e):
                # Fallback: use predict_proba with KernelExplainer for XGBoost multiclass
                print(f"Warning: TreeExplainer failed for {type(model).__name__}. Using KernelExplainer as fallback.")
                background = shap.sample(X_df, min(100, len(X_df)), random_state=42 + iteration)
                explainer = shap.KernelExplainer(model.predict_proba, background)
                shap_values = explainer.shap_values(background, nsamples=100)
                X_df_used = background
                
                # Skip to plotting
                fig = plt.figure(figsize=(12, 6))
                shap.summary_plot(shap_values, X_df_used, feature_names=feature_names,
                                 plot_type="bar", show=False)
                plt.title(title, fontsize=14, pad=20)
                plt.xlabel("Mean |SHAP value|", fontsize=12)
                if path:
                    plt.savefig(f"{path}_bar_{iteration}.png", dpi=300, bbox_inches='tight')
                plt.tight_layout()
                return shap_values, explainer, fig, X_df_used
            else:
                raise
        
        X_df_used = X_df
    else:  # kernel
        # Usar muestra para KernelExplainer (más rápido)
        background = shap.sample(X_df, min(100, len(X_df)), random_state=42 + iteration)
        explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_values = explainer.shap_values(background, nsamples=100)
        X_df_used = background

    # Gráfico de barras
    fig = plt.figure(figsize=(12, 6))
    shap.summary_plot(shap_values, X_df_used, feature_names=feature_names,
                     plot_type="bar", show=False)
    plt.title(title, fontsize=14, pad=20)
    plt.xlabel("Mean |SHAP value|", fontsize=12)
    if path:
        plt.savefig(f"{path}_bar_{iteration}.png", dpi=300, bbox_inches='tight')
    plt.tight_layout()

    return shap_values, explainer, fig, X_df_used

def extract_common_values_from_csv(csv_path):
    """
    Lee un CSV y extrae los valores comunes en todas las columnas,
    ordenados por la suma de sus posiciones (ranking) en cada columna.
    
    Cada valor en una columna recibe un puntaje igual a su índice de fila
    (posición 0 = más importante). Para los valores comunes a todas las columnas,
    se suman sus posiciones y se ordenan de menor a mayor (mejor ranking global primero).
    
    Args:
        csv_path (str): Ruta al archivo CSV
        
    Returns:
        list: Lista de valores comunes ordenados por ranking agregado (menor suma = mejor)
    """
    df = pd.read_csv(csv_path)
    
    # Para cada columna, construir un dict {valor: posición}
    column_rankings = []
    column_sets = []
    for col in df.columns:
        rank_dict = {}
        for pos, val in enumerate(df[col]):
            if pd.notna(val) and str(val).strip() != '':
                rank_dict[val] = pos
        column_rankings.append(rank_dict)
        column_sets.append(set(rank_dict.keys()))
    
    if not column_sets:
        return []
    
    # Valores comunes en todas las columnas
    common_values = set.intersection(*column_sets)
    
    if not common_values:
        return []
    
    # Calcular la suma de posiciones para cada valor común
    scores = {}
    for val in common_values:
        scores[val] = sum(rank_dict[val] for rank_dict in column_rankings) # iterar en cada diccionario de cada columna
    
    # Ordenar por puntaje agregado (menor suma = mejor ranking global)
    sorted_values = sorted(scores.keys(), key=lambda v: scores[v])
    
    return sorted_values

def extract_frequent_values_from_csv(csv_path, threshold_percentage=80):
    """
    Lee un CSV y extrae los valores que aparecen en al menos el threshold_percentage
    de las columnas, ordenados por la suma de sus posiciones (ranking) en cada columna.
    
    A diferencia de extract_common_values_from_csv que requiere que los valores aparezcan
    en TODAS las columnas, esta función permite un umbral configurable.
    
    Cada valor en una columna recibe un puntaje igual a su índice de fila
    (posición 0 = más importante). Para los valores que aparecen en al menos el
    threshold_percentage de columnas, se suman sus posiciones y se ordenan de
    menor a mayor (mejor ranking global primero).
    
    Args:
        csv_path (str): Ruta al archivo CSV
        threshold_percentage (float): Porcentaje mínimo de columnas en las que debe 
                                     aparecer un valor (default: 80)
        
    Returns:
        list: Lista de valores frecuentes ordenados por ranking agregado (menor suma = mejor)
    """
    df = pd.read_csv(csv_path)
    
    # Para cada columna, construir un dict {valor: posición}
    column_rankings = []
    column_sets = []
    for col in df.columns:
        rank_dict = {}
        for pos, val in enumerate(df[col]):
            if pd.notna(val) and str(val).strip() != '':
                rank_dict[val] = pos
        column_rankings.append(rank_dict)
        column_sets.append(set(rank_dict.keys()))
    
    if not column_sets:
        return []
    
    # Calcular cuántas columnas mínimas debe aparecer un valor
    num_columns = len(column_sets)
    min_appearances = int(np.ceil(num_columns * threshold_percentage / 100))
    
    # Obtener todos los valores únicos
    all_values = set()
    for col_set in column_sets:
        all_values.update(col_set)
    
    # Filtrar valores que aparecen en al menos threshold_percentage de columnas
    frequent_values = set()
    for val in all_values:
        count = sum(1 for col_set in column_sets if val in col_set)
        if count >= min_appearances:
            frequent_values.add(val)
    
    if not frequent_values:
        return []
    
    # Calcular la suma de posiciones para cada valor frecuente
    # Solo sumamos las posiciones en las columnas donde el valor aparece
    scores = {}
    for val in frequent_values:
        scores[val] = sum(rank_dict[val] for rank_dict in column_rankings if val in rank_dict)
    
    # Ordenar por puntaje agregado (menor suma = mejor ranking global)
    sorted_values = sorted(scores.keys(), key=lambda v: scores[v])
    
    return sorted_values


# =========== LOAD DATA =================

df_imputed = pd.read_csv(CFG.path_df_imputed_corrected)
df_imputed.columns = clean_feature_names(df_imputed.columns)

if CFG.cuartiles_train:
    n_clases = 2
else:
    n_clases = 3

# Preparar datos
X_train, X_test, y_train, y_test, feature_names, class_dist = preparar_datos(
    df_imputed, n_clases=n_clases, element="Nitrogen", best_variables=None, CFG=CFG, random_state=BEST_ITER+41
)


NUMBER_ITERATIONS = 10
results_iteration_shap = {}
for i in range(1, NUMBER_ITERATIONS +1):
    for algorithm, value in all_results.items():
        counter = 0
        for modelo in value:
            
            print(f"Iteración {i+1} - Algoritmo: {algorithm}")
            if 'grid_search' in modelo:
                pipeline = modelo['grid_search'].best_estimator_
            else:
                pipeline = modelo['best_model']
            output_dir = os.path.join(os.getcwd(), "shap_outputs", model_name, str(i))
            os.makedirs(output_dir, exist_ok=True)

            scaler = pipeline.named_steps['scaler']
            clf = pipeline.named_steps['clf']
            X_test_scaled = scaler.transform(X_test)

            shap_values, explainer, fig, X_df_used = plot_shap_importance(
                clf, X_test_scaled, feature_names, model_type=MODELS_CONFIG[algorithm]['model_type'], n_clases=2,
                title=f"SHAP Feature Importance - {algorithm} Iteration {i+1}",
                path=os.path.join(output_dir, f"shap_importance_{algorithm}_iteration_{i+1}"),
                iteration=i
            )
            plt.close(fig)  # Cerrar figura para liberar memoria
            # guardar shap_values y explainer
            all_results[algorithm][counter]['shap_values'] = shap_values
            all_results[algorithm][counter]['explainer'] = explainer
            all_results[algorithm][counter]['X_scaled_df'] = X_df_used.copy()
            counter += 1
    
    best_x_percentage_all_algorithms = extract_top_x_percent_features(all_results, percent=0.8, class_path=output_dir, CFG=CFG)
    results_iteration_shap[i] = best_x_percentage_all_algorithms

output_dir = os.path.join(os.getcwd(), "shap_outputs", model_name)
# guardar results_iteration_shap como json

with open(os.path.join(output_dir, f"results_iteration_shap.json"), 'w') as f:
    json.dump(results_iteration_shap, f, indent=4)

with open(os.path.join(output_dir, f"results_iteration_shap.json"), 'r') as f:
    results_iteration_shap = json.load(f)
# guardar results_iteration_shap en diferentes CSV por categoría y algoritmo
# Se debe guardar como output_dir/model/vars_{alg_name}.csv
# Cada columna del csv es una iteración de ese algoritmo
models = results_iteration_shap['5'].keys()  # Obtener nombre modelo (ej: Nitrogen, Phosphorus, Potassium)
for model_name in models:
    model_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    for iteration in results_iteration_shap.keys():
        algs = results_iteration_shap[iteration][model_name].keys()  # Obtener nombre algoritmo (ej: RF, SVM, MLP)
        for alg_name in algs:
            features = results_iteration_shap[iteration][model_name][alg_name]
            df_features = pd.DataFrame({f"Iteration_{int(iteration)+1}": features})
            csv_path = os.path.join(model_dir, f"vars_{alg_name}.csv")
            if os.path.exists(csv_path):
                df_existing = pd.read_csv(csv_path)
                df_combined = pd.concat([df_existing, df_features], axis=1)
                df_combined.to_csv(csv_path, index=False)
            else:
                df_features.to_csv(csv_path, index=False)

# Extraer y guardar variables comunes para cada algoritmo
for model in models:
    model_dir = os.path.join(output_dir, model)
    if os.path.exists(model_dir):
        csv_files = [f for f in os.listdir(model_dir) if f.endswith('.csv')]
        
        
        # Diccionario para almacenar variables comunes por algoritmo
        common_vars_dict = {}
        
        for csv_file in csv_files:
            csv_path = os.path.join(model_dir, csv_file)
            if "common" not in csv_path:
                common_vars = extract_frequent_values_from_csv(csv_path, threshold_percentage=80)
                
                algorithm_name = csv_file.replace('vars_', '').replace('.csv', '')

                
                # Agregar al diccionario
                common_vars_dict[algorithm_name] = common_vars
        
        # Guardar como CSV
        if common_vars_dict:
            # Convertir a DataFrame (maneja listas de diferentes longitudes con NaN)
            df_common = pd.DataFrame.from_dict(common_vars_dict, orient='index').T
            
            # Guardar CSV
            output_csv = os.path.join(model_dir, f"common_vars_{model}_80.csv")
            df_common.to_csv(output_csv, index=False)
            print(f"\nGuardado: {output_csv}")

common_all_models = {} # para almacenar las variables comunes en todos los modelso. e.g en Nitrogen, Phosphorus, Potassium
for model in models:
    model_dir = os.path.join(output_dir, model)
    common_csv_path = os.path.join(model_dir, f"common_vars_{model}_80.csv")
    if os.path.exists(common_csv_path):
        th_percentage_final = 80 # Variables presentes en el 80% de algoritmos (en este caso 4/5 algoritmos)
        common_vars = extract_frequent_values_from_csv(common_csv_path, threshold_percentage=th_percentage_final) 
        
        print(common_vars)
        #guardar common_vars
        common_all_models[model] = common_vars

# guardar en csv
df_common_all_models = pd.DataFrame.from_dict(common_all_models, orient='index').T
df_common_all_models.to_csv(os.path.join(output_dir, f"common_vars_all_models_{th_percentage_final}.csv"), index=False)



#===========================================================
#========== PERMUTATION IMPORTANCE ==========================
#==============================================================

# Recalcular class_path segun flags actuales.
if CFG.include_prod:
    if CFG.cuartiles_train:
        CFG.class_path = f"{CFG.Root}/Resultados/classification_cuartiles_include_prod/"
    else:
        CFG.class_path = f"{CFG.Root}/Resultados/classification_include_prod/"
else:
    if CFG.cuartiles_train:
        CFG.class_path = f"{CFG.Root}/Resultados/classification_cuartiles_exclude_prod/"
    else:
        CFG.class_path = f"{CFG.Root}/Resultados/classification_exclude_prod/"

# ======= Permutation Importance ==========================
if CFG.individual_train and not CFG.cuartiles_train:
    dir_path_permutation = f"{CFG.class_path}permutation_importance/"
    permutation_importance_NPK(all_results, df_imputed, dir_path_permutation, CFG.class_path, CFG=CFG, random_state = BEST_ITER+41)

if CFG.cuartiles_train:
    dir_path_permutation = f"{CFG.class_path}permutation_importance/"
    permutation_importance_Quartiles(all_results, df_imputed, dir_path_permutation, CFG.class_path, CFG=CFG)
