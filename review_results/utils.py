import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import seaborn as sns
import shap
from itertools import product
import h5py
import io
from PIL import Image

# Sklearn imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split, cross_validate
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score, precision_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.decomposition import PCA
from sklearn.base import clone

# Imbalanced-learn imports
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from xgboost import XGBClassifier

import pickle
import json
import os


import eli5
from eli5.sklearn import PermutationImportance

import warnings
warnings.filterwarnings('ignore')

from config import *


# ======================================
# ====== CODIFICACIÓN DE CLASES ========
# ====================================== 

def codificar_clase(n, p, k):
    """Codificación base de 8 clases basada en combinaciones de nutrientes."""
    if n == 1 and p == 1 and k == 1:
        return 7
    elif n == 1 and p == 1:
        return 4
    elif n == 1 and k == 1:
        return 5
    elif p == 1 and k == 1:
        return 6
    elif n == 1:
        return 1
    elif p == 1:
        return 2
    elif k == 1:
        return 3
    else:
        return 0

def codificar_clase_personalizada(n, p, k, n_clases):
    """Codificación personalizada para 2-5 clases."""
    clase_base = codificar_clase(n, p, k)
    if n_clases == 2:
        return 1 if clase_base in [4, 5, 6, 7] else 0
    elif n_clases == 3:
        if clase_base in [1, 2, 3]:
            return 1
        elif clase_base in [4, 5, 6, 7]:
            return 2
        else:
            return 0
    elif n_clases == 4:
        return sum([n == 1, p == 1, k == 1])
    elif n_clases == 5:
        if n == 1 and p == 1 and k == 1:
            return 3
        elif (n == 1 and p == 1) or (n == 1 and k == 1) or (p == 1 and k == 1):
            return 2
        elif n == 1 or p == 1 or k == 1:
            return 1
        elif n == 0 and p == 0 and k == 0:
            return 0
        else:
            return 4
    else:
        return clase_base

def codificar_clase_6(c):
    """Agrupación para 6 clases."""
    if c == 0:
        return 0
    elif c == 1:
        return 1
    elif c in [2, 3]:
        return 2
    elif c in [4, 5]:
        return 3
    elif c == 6:
        return 4
    elif c == 7:
        return 5

def codificar_clase_7(c):
    """Agrupación para 7 clases."""
    if c in [5, 6]:
        return 5
    elif c == 7:
        return 6
    else:
        return c

def codificar_clase_nk_9(n, k):
    """Codificación de 9 clases basada en N y K (ignora P)."""
    if n not in [0, 1, 2] or k not in [0, 1, 2]:
        return None
    return 3 * n + k

def codificar_clase_agrupada(n, p, k, n_clases):
    """Función principal de codificación según número de clases."""
    base = codificar_clase(n, p, k)
    if n_clases == 5:
        return codificar_clase_personalizada(n, p, k, 5)
    elif n_clases == 6:
        return codificar_clase_6(base)
    elif n_clases == 7:
        return codificar_clase_7(base)
    elif n_clases == 8:
        return base
    elif n_clases == 9:
        return codificar_clase_nk_9(n, k)
    else:
        return codificar_clase_personalizada(n, p, k, n_clases)

def codificar_clase_individual(df, element, CFG):
    if element not in CFG.elements_list:
        raise ValueError(f"Elemento '{element}' no válido. Debe ser uno de {CFG.elements_list}.")
    return df[element].copy()

def codificar_clase_cuartiles(df, filter_data=True, CFG=None):
    """
    Codifica clases basado en cuartiles de productividad.
    
    Args:
        df: DataFrame con los datos
        filter_data: Si True, elimina tratamientos que no están en Q1 o Q4
        CFG: Configuration object
    
    Returns:
        Series con las clases codificadas (y DataFrame filtrado si filter_data=True)
    """
    # Leer json con tratamientos asociados a cuartiles
    with open(CFG.treat_quantiles_path, 'r') as f:
        treatments_quantile_unified = json.load(f)
    
    # Crear diccionario inverso: {treatment_num: clase}
    treatment_to_class = {}
    for clase, treatments in treatments_quantile_unified.items():
        for treatment in treatments:
            treatment_to_class[treatment] = int(clase)
    
    if filter_data:
        # Filtrar solo tratamientos Q1 y Q4
        valid_treatments = list(treatment_to_class.keys())
        df_filtered = df[df['Treatment_Num'].isin(valid_treatments)].copy()
        
        # Mapear a clases y remapear a 0 y 1
        y = df_filtered['Treatment_Num'].map(treatment_to_class)
        
        return df_filtered, y
    else:
        # Solo mapear sin filtrar (generará NaN)
        y = df['Treatment_Num'].map(treatment_to_class)
        return y

def clean_feature_names(feature_names):
    """Remove characters that XGBoost doesn't allow in feature names."""
    cleaned = []
    for name in feature_names:
        # Replace brackets with parentheses or underscores
        cleaned_name = name.replace('[', '(').replace(']', ')').replace('<', '_lt_')
        cleaned.append(cleaned_name)
    return cleaned
# ======================================
# ====== Plot Confusion Matrix ========
# ======================================
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Greens):
    """
    This function prints and plots the confusion matrix.
    Args:
        cm: Confusion matrix (array)
        classes: List of class names
        normalize: If True, normalize the confusion matrix
        title: Title of the plot
        cmap: Color map for the plot
    Returns:
        fig: Matplotlib figure object
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.set_title(title, fontsize=14, pad=20)
    plt.colorbar(im, ax=ax)

    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.set_yticklabels(classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], fmt),
               horizontalalignment="center",
               color="white" if cm[i, j] > thresh else "black")

    ax.set_ylabel('True label', fontsize=12)
    ax.set_xlabel('Predicted label', fontsize=12)
    plt.tight_layout()
    return fig

# =====================================
# ============ Prepare Data ============
# ====================================

def preparar_datos(df_imputed, n_clases=3, element="Nitrogen", test_size=0.3, random_state=42,
                   best_variables = None, CFG=None):
    """Prepara los datos para entrenamiento.

    Args:
        df_imputed (DataFrame): DataFrame con datos imputados.
        n_clases (int): Número de clases para la codificación.
        element (str): Elemento a utilizar para la codificación individual.
        test_size (float): Proporción del conjunto de prueba.
        random_state (int): Semilla para reproducibilidad.
        best_variables (list): Lista de variables mas importantes para realizar el entrenamiento
                            solo con esas vairables y comparar. Si se pasa la lista, se prodcede
                            a eliminar(drop) las otras variables.
        CFG: Configuration object
    Returns:
        tuple: (X_train, X_test, y_train, y_test, feature_names, class_distribution)
    """
    df = df_imputed.copy()
    
    # NOTE: Ahroa se trabajan con 3 clases
    if CFG.individual_train:
        df['target'] = codificar_clase_individual(df, element=element, CFG=CFG)
    elif CFG.cuartiles_train:
        df, df['target'] = codificar_clase_cuartiles(df, filter_data=True, CFG=CFG)
    else:
    # NOTE: La siguiente codificación se hacía cuando se trataban 2 a 9 clases
        df['target'] = df.apply(
            lambda row: codificar_clase_agrupada(
                row['Nitrogen'], row['Phosphorus'], row['Potassium'], n_clases
            ), axis=1
        )
    

    # Distribución de clases
    class_distribution = df['target'].value_counts().sort_index()

    # Eliminar columnas no necesarias
    columns_to_drop = ['Nitrogen', 'Phosphorus', 'Potassium', 'target',
                       'Clase_custom', 'Treatment_Num', 'Year', 'Month', 'Day']
    # Eliminar variables de productividad si no se desean incluir
    if not CFG.include_prod:
        columns_to_drop += CFG.productivity_vars
    if best_variables != None:
        for var_name in df.columns:
            if var_name not in best_variables:
                columns_to_drop.append(var_name)

    X = df.drop(columns=columns_to_drop, errors='ignore')
    y = df['target']

    # División de datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test, X.columns, class_distribution

def extract_X_y_classification(df_imputed, n_clases=3, element="Nitrogen", best_variables = None, CFG=None):
    """
    Extrae las características (X) y las etiquetas (y) del DataFrame para clasificación
    NO separa en train y test.

    Args:
        df_imputed (DataFrame): DataFrame con datos imputados.
        n_clases (int): Número de clases para la codificación.
        element (str): Elemento a utilizar para la codificación individual.
        best_variables (list): Lista de variables mas importantes para realizar el entrenamiento
        CFG: Configuration object
    Returns:
        :X (DataFrame): Características para el modelo.
        :y (Series): Etiquetas de clase.
        :class_distribution (Series): Distribución de clases en y.
    """
    df = df_imputed.copy()
    
    # NOTE: Ahroa se trabajan con 3 clases
    if CFG.individual_train:
        df['target'] = codificar_clase_individual(df, element=element, CFG=CFG)
    elif CFG.cuartiles_train:
        df, df['target'] = codificar_clase_cuartiles(df, filter_data=True, CFG=CFG)
    else:
    # NOTE: La siguiente codificación se hacía cuando se trataban 2 a 9 clases
        df['target'] = df.apply(
            lambda row: codificar_clase_agrupada(
                row['Nitrogen'], row['Phosphorus'], row['Potassium'], n_clases
            ), axis=1
        )
    

    # Distribución de clases
    class_distribution = df['target'].value_counts().sort_index()

    # Eliminar columnas no necesarias
    columns_to_drop = ['Nitrogen', 'Phosphorus', 'Potassium', 'target',
                       'Clase_custom', 'Treatment_Num', 'Year', 'Month', 'Day']
    # Eliminar variables de productividad si no se desean incluir
    if not CFG.include_prod:
        columns_to_drop += CFG.productivity_vars
    if best_variables != None:
        for var_name in df.columns:
            if var_name not in best_variables:
                columns_to_drop.append(var_name)
    X = df.drop(columns=columns_to_drop, errors='ignore')
    y = df['target']

    return X, y, X.columns, class_distribution

def calcuate_PCA(X, n_components=2):
    '''
    Calcula PCA para reducción de dimensionalidad.
    Args:
        X (DataFrame): Datos de entrada.
        n_components (int): Número de componentes principales.
    Returns:
        tuple: (X_pca, pca) donde X_pca son los datos transformados y pca es el objeto PCA.
    '''
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    return X_pca, pca
# ==================================================
# ====== Plot and save SHAP Importance ==============
# =================================================
def plot_shap_importance(model, X_test, feature_names, model_type='tree', n_clases=2, title="SHAP Feature Importance", path=None):
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
        # Fix for XGBoost multi-class compatibility with SHAP
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
                background = shap.sample(X_df, min(100, len(X_df)), random_state=42)
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
                    plt.savefig(f"{path}_bar.png", dpi=300, bbox_inches='tight')
                plt.tight_layout()
                return shap_values, explainer, fig, X_df_used
            else:
                raise
        
        X_df_used = X_df
    else:  # kernel
        # Usar muestra para KernelExplainer (más rápido)
        background = shap.sample(X_df, min(100, len(X_df)), random_state=42)
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
        plt.savefig(f"{path}_bar.png", dpi=300, bbox_inches='tight')
    plt.tight_layout()

    return shap_values, explainer, fig, X_df_used
# def plot_shap_importance(model, X_test, feature_names, model_type='tree', n_clases=2, title="SHAP Feature Importance", path=None):
#     """Genera gráficos de importancia SHAP.

#     Args:
#         model: Modelo entrenado.
#         X_test (DataFrame): Datos de prueba.
#         feature_names (list): Nombres de las características.
#         model_type (str): 'tree' para TreeExplainer, 'kernel' para KernelExplainer.
#                          'kernel' se utiliza para modelos no basados en árboles como SVM, KNN, MLP.
#         n_clases (int): Número de clases en el modelo.
#         title (str): Título del gráfico.
#         path (str): Ruta para guardar el gráfico (sin extensión).
#     Returns:
#         tuple: (shap_values, explainer, shap_fig, X_df_used)

#     """
#     X_df = pd.DataFrame(X_test, columns=feature_names)

#     if model_type == 'tree':
#         explainer = shap.TreeExplainer(model)
#         shap_values = explainer.shap_values(X_df)
#         X_df_used = X_df
#     else:  # kernel
#         # Usar muestra para KernelExplainer (más rápido)
#         background = shap.sample(X_df, min(100, len(X_df)), random_state=42)
#         explainer = shap.KernelExplainer(model.predict_proba, background)
#         shap_values = explainer.shap_values(background, nsamples=100)
#         X_df_used = background

#     # Gráfico de barras
#     fig = plt.figure(figsize=(12, 6))
#     shap.summary_plot(shap_values, X_df_used, feature_names=feature_names,
#                      plot_type="bar", show=False)
#     plt.title(title, fontsize=14, pad=20)
#     plt.xlabel("Mean |SHAP value|", fontsize=12)
#     if path:
#         plt.savefig(f"{path}_bar.png", dpi=300, bbox_inches='tight')
#     plt.tight_layout()

#     return shap_values, explainer, fig, X_df_used

# ======================================
# ===== Auxiliar Train Functions ============
# ======================================
#
def initialize_classification_results():
    results = {}
    results["model_params"] = {}
    results["acc"] = {"train": [], "test": []}
    results["precision"] = {"train": [], "test": []}
    results["recall"] = {"train": [], "test": []}
    results["f1_score"] = {"train": [], "test": []}
    return results

def calculate_classification_metrics(y_true, y_pred, average='weighted'):
    """
    Calcula métricas de clasificación
    
    Args:
        y_true: Etiquetas verdaderas
        y_pred: Etiquetas predichas
        average: Tipo de promedio para métricas multiclase
    
    Returns:
        tuple: (accuracy, precision, recall, f1_score)
    """
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=average, zero_division=0)
    recall = recall_score(y_true, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
    return acc, precision, recall, f1

def accumulate_classification_results(results, tipo="train", acc=None, precision=None, recall=None, f1=None):
    results["acc"][tipo].append(acc)
    results["precision"][tipo].append(precision)
    results["recall"][tipo].append(recall)
    results["f1_score"][tipo].append(f1)

def print_classification_report(model_name, n_clases, acc_train, acc_test, f1_train, f1_test, best_params, class_dist):
    print(f"\n{'='*60}")
    print(f"  {model_name} - {n_clases} CLASSES")
    print(f"{'='*60}")
    print(f"Best parameters: {best_params}")
    print(f"\nTrain Performance:")
    print(f"  Accuracy: {acc_train:.4f}")
    print(f"  F1 Macro: {f1_train:.4f}")
    print(f"\nTest Performance:")
    print(f"  Accuracy: {acc_test:.4f}")
    print(f"  F1 Macro: {f1_test:.4f}")
    print(f"\nClass Distribution:")
    print(class_dist)

def return_classification_metrics(y_test, y_test_pred):
    acc_test = accuracy_score(y_test, y_test_pred)
    f1_test = f1_score(y_test, y_test_pred, average='micro')
    return acc_test, f1_test

def find_best_model_params(results):
    """
    Encontrar los parametros del modelo que dio mejor resultado en test f1_score
    Args:
        results (dict): Diccionario con resultados acumulados.
    Returns:
        dict: Parámetros del mejor modelo.
    """
    best_f1 = -1
    best_params = None
    for i in range(len(results['f1_score']['test'])):
        if results['f1_score']['test'][i] > best_f1:
            best_f1 = results['f1_score']['test'][i]
            best_params = results['model_params'][i]
    return best_params

def strip_pipeline_prefix(params):
    """
    Remove pipeline prefixes (like 'clf__') from parameter names.
    Args:
        params (dict): Parameters with pipeline prefixes.
    Returns:
        dict: Parameters with prefixes removed.
    """
    stripped = {}
    for key, value in params.items():
        # Remove any prefix ending with '__'
        if '__' in key:
            new_key = key.split('__', 1)[-1]  # Take everything after the first '__'
            stripped[new_key] = value
        else:
            stripped[key] = value
    return stripped


# ====================================================
# ========= CLASSIFICATION TRAIN FUNCTIONS ===========
# ====================================================
def train_test_class_nested(df_imputed, n_clases, model_name, model_config, element = "Nitrogen",
                              usar_smote=True, mostrar_graficos=True, calcular_shap=True,
                              h5_file=None,
                              dir_path= "../",
                              best_variables = None, train_pca = False, n_components = None, CFG=None):
    '''
    Función para el entrenamiento de modelos de clasificación utilizando cross-validation anidada.
    El objetivo es reducir overfitting. Toma más tiempo.
    Para extraer los valores SHAP (y pensando en una forma de producción), 
    que originialmente necesitaban los datos de train, ahora es necesario
    obtener el mejor modelo de los que se entrenaron en cada K-Fold exterior, y posteriormente
    volver a entrenar con TODOS los datos de train iniciales. Tener precaución.
    
    '''
    # Preparar datos
    X, y, feature_names, class_distribution = extract_X_y_classification(
        df_imputed, n_clases=n_clases, element=element, CFG=CFG
    )
    if train_pca:
        # Aplicar PCA
        X, pca = calcuate_PCA(X, n_components=n_components)
    
    skf_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    skf_inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    results = initialize_classification_results()
    
    for fold_idx, (train_index, val_index) in enumerate(skf_outer.split(X, y)):
        X_train, X_test = X.iloc[train_index], X.iloc[val_index]
        y_train, y_test = y.iloc[train_index], y.iloc[val_index]

        # Pipeline (evita data leakage)
        if usar_smote:
            pipeline = ImbPipeline([
                ('scaler', StandardScaler()),
                ('smote', SMOTE(random_state=42)),
                ('clf', model_config['estimator'])
            ])
        else:
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('clf', model_config['estimator'])
            ])
        
        # GridSearchCV con CV interno
        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=model_config['param_grid'],
            cv=skf_inner,
            scoring='f1_micro',
            n_jobs=-1,
            verbose=1,
            return_train_score=True
        )
        
        grid.fit(X_train, y_train)
        
        # Extraer métricas del INNER CV
        best_idx = grid.best_index_
        inner_train_f1 = grid.cv_results_['mean_train_score'][best_idx]        
        # 8. Evaluar en outer fold
        y_pred_test = grid.predict(X_test)
        
        # Usar tus funciones existentes
        
        #avg = "weighted" if n_clases != None else "micro" 
        avg = "micro"
        test_acc, test_prec, test_rec, test_f1 = calculate_classification_metrics(
            y_test, y_pred_test, average=avg
        )
        
        # Acumular con tu función (solo test, train viene del CV interno)
        accumulate_classification_results(
            results, "train", inner_train_f1, None, None, inner_train_f1
        )
        accumulate_classification_results(
            results, "test", test_acc, test_prec, test_rec, test_f1
        )
        
        # Guardar parámetros
        results["model_params"][fold_idx] = grid.best_params_
    
    
    if CFG.individual_train:
        model_path =  f"{dir_path}/models/{model_name.replace(' ', '_')}_nclases_{3}_{element}_params.pkl"
    elif CFG.cuartiles_train:
        model_path =  f"{dir_path}/models/{model_name.replace(' ', '_')}_nclases_{2}_cuartiles_params.pkl"
    else:
        model_path = f"{dir_path}/models/{model_name.replace(' ', '_')}_nclases_{n_clases}_params.pkl"
    os.makedirs(f"{dir_path}/models", exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(results['model_params'], f)
    
    # Encontrar cual fue el mejor modelo de los k-folds (el que tuvo mejor f1_score en test) 
    # Y se vuelve a entrenar para obtener los valores SHAP con el modelo final entrenado con TODOS los datos de train (no solo el fold)
    best_params = find_best_model_params(results)
    # Strip the 'clf__' prefix from parameters for direct estimator use
    estimator_params = strip_pipeline_prefix(best_params)
    

    # Preparar datos completos
    X_train, X_test, y_train, y_test, feature_names, class_dist = preparar_datos(
        df_imputed, n_clases, element=element, best_variables=best_variables, CFG=CFG
    )
    if train_pca:
        # Aplicar PCA
        X_train, pca = calcuate_PCA(X_train, n_components=n_components)
        X_test = pca.transform(X_test)
        
    # Entrenar modelo FINAL con TODOS los datos
    if usar_smote:
        final_pipeline = ImbPipeline([
            ('scaler', StandardScaler()),
            ('smote', SMOTE(random_state=42)), # TODO: Ver si es correcto usar SMOTE aquí
            ('clf', clone(model_config['estimator']).set_params(**estimator_params))
        ])
    else:
        final_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', clone(model_config['estimator']).set_params(**estimator_params))
        ])
    
    final_pipeline.fit(X_train, y_train)
    
    # Guardar PIPELINE completo
    if CFG.individual_train:
        model_path = f"{dir_path}/models/{model_name.replace(' ', '_')}_nclases_3_{element}.pkl"
    elif CFG.cuartiles_train:
        model_path = f"{dir_path}/models/{model_name.replace(' ', '_')}_nclases_2_cuartiles.pkl"
    else:
        model_path = f"{dir_path}/models/{model_name.replace(' ', '_')}_nclases_{n_clases}.pkl"
    
    os.makedirs(f"{dir_path}/models", exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(final_pipeline, f)
    
    print(f"Model saved: {model_path}")

    # SHAP
    shap_values = None
    X_scaled_df = None
    if calcular_shap:
        
        # Preparar datos completos
        # X_train, X_test, y_train, y_test, feature_names, class_dist = preparar_datos(
        #     df_imputed, n_clases, element=element, best_variables=best_variables
        # )
        
        
        # Preprocesar
        clf = final_pipeline.named_steps['clf']
        scaler = final_pipeline.named_steps['scaler']
        X_scaled = scaler.transform(X_test)
        
        path_images_shap = f"{dir_path}/{model_name.replace(' ', '_')}/"
        os.makedirs(path_images_shap, exist_ok=True)
        
        shap_values, _, fig_shap, X_scaled_df = plot_shap_importance(
            clf,  # Modelo final
            X_scaled,
            feature_names,
            model_type=model_config['model_type'],
            n_clases=n_clases,
            title=f"SHAP Feature Importance - {model_name} ({n_clases} classes)",
            path=f"{path_images_shap}shap"
        )

        if isinstance(shap_values, list):
            # Caso típico multiclase con lista
            for i in range(len(shap_values)):
                print(f"Clase {i} - SHAP summary")
                # Crear figura nueva
                plt.figure()
                shap.summary_plot(shap_values[i], X_scaled_df, plot_type="dot")
        elif len(shap_values.shape) == 2:
            # Caso binario: (n_samples, n_features)
            print("SHAP summary - Binary Classification")
            plt.figure()
            shap.summary_plot(shap_values, X_scaled_df, plot_type="dot")
        else:
            # Caso de array de 3D: (n_samples, n_features, n_classes)
            num_clases = shap_values.shape[-1]
            for i in range(num_clases):
                print(f"Clase {i} - SHAP summary")
                shap.summary_plot(shap_values[..., i], X_scaled_df, plot_type="dot")

        if mostrar_graficos:
            plt.show()

    if CFG.individual_train:
        n_clases_str = f"{3}_{element}"
    elif CFG.cuartiles_train:
        n_clases_str = "2_Quartiles"
    else:
        n_clases_str = str(n_clases)
    resultados_finales = {
        'n_clases': n_clases_str,
        'model_name': model_name, # Nombre del algoritmo
        'accuracy_train': np.mean(results['acc']['train']),
        'accuracy_test': np.mean(results['acc']['test']),
        'f1_train': np.mean(results['f1_score']['train']),
        'f1_test': np.mean(results['f1_score']['test']),
        'accuracy_test_std': np.std(results['acc']['test']),
        'f1_test_std': np.std(results['f1_score']['test']),
    
        'acc_raw' : results['acc'],
        'precision_raw' : results['precision'],
        'best_params': results['model_params'],
        'class_distribution': class_distribution,
        # 'confusion_matrix_train': cm_train,
        # 'confusion_matrix_test': cm_test,
        'results_CV': results,
        'grid_search': results['model_params'],
        'shap_values': shap_values,
        'X_scaled_df': X_scaled_df
    }

    return resultados_finales

def read_best_variables(json_file):
    with open(json_file, 'r') as file:
        model_variables = json.load(file)
    best_variables = []

    for model in model_variables:
        for variable in model_variables[model]:
            if variable not in best_variables:
                best_variables.append(variable)
    return best_variables



def build_pipeline(model_config, usar_smote = False):
    if usar_smote:
        pipeline_steps = [
            ('scaler', StandardScaler()),
            ('smote', SMOTE(random_state=42)),
            ('clf', clone(model_config['estimator']))
        ]
        pipe = ImbPipeline(pipeline_steps)
    else:
        pipeline_steps = [
            ('scaler', StandardScaler()),
            ('clf', clone(model_config['estimator']))
        ]
        pipe = Pipeline(pipeline_steps)
    return pipe

# TODO: Revisar si utilizar X_test en vez de X_train
def calculate_shap(model, X_test, dir_path, model_name, feature_names, model_config, n_clases,
                   mostrar_graficos=False):
    # Check if it's a GridSearchCV or a Pipeline
    if hasattr(model, 'best_estimator_'):
        # It's a GridSearchCV object
        pipeline = model.best_estimator_
    else:
        # It's already a Pipeline
        pipeline = model
    
    # Extract scaler and clf from the pipeline
    scaler = pipeline.named_steps['scaler']
    clf = pipeline.named_steps['clf']
    X_test_scaled = scaler.transform(X_test)
    #revisar si existe la carpeta
    path_images_shap = f"{dir_path}/{model_name.replace(' ', '_')}/"
    os.makedirs(path_images_shap, exist_ok=True)

    shap_values, _, fig_shap, X_scaled_df = plot_shap_importance(
        clf, X_test_scaled, feature_names,
        model_type=model_config['model_type'],
        n_clases=n_clases,
        title=f"SHAP Feature Importance - {model_name} ({n_clases} classes)",
        path=f"{path_images_shap}{n_clases}_clases_shap"
    )

    if isinstance(shap_values, list):
        # Caso típico multiclase con lista
        for i in range(len(shap_values)):
            print(f"Clase {i} - SHAP summary")
            # Crear figura nueva
            plt.figure()
            shap.summary_plot(shap_values[i], X_scaled_df, plot_type="dot")
    elif len(shap_values.shape) == 2:
        # Caso binario: (n_samples, n_features)
        print("SHAP summary - Binary Classification")
        plt.figure()
        shap.summary_plot(shap_values, X_scaled_df, plot_type="dot")
    else:
        # Caso de array de 3D: (n_samples, n_features, n_classes)
        num_clases = shap_values.shape[-1]
        for i in range(num_clases):
            print(f"Clase {i} - SHAP summary")
            shap.summary_plot(shap_values[..., i], X_scaled_df, plot_type="dot")

    if mostrar_graficos:
        plt.show()
    return shap_values, X_scaled_df

def train_test_model(df_imputed, n_clases, model_name, model_config, element = "Nitrogen",
                              usar_smote=True, mostrar_graficos=True, calcular_shap=True,
                              h5_file=None,
                              dir_path= "../",
                              best_variables = None, train_pca = False, n_components = None,
                              CFG=None):
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
    # Preparar datos
    X_train, X_test, y_train, y_test, feature_names, class_dist = preparar_datos(
        df_imputed, n_clases, element=element, best_variables=best_variables, CFG=CFG
    )

    if train_pca:
        # Aplicar PCA
        X_train, pca = calcuate_PCA(X_train, n_components=n_components)
        X_test = pca.transform(X_test)

    # Construir objeto de pipeline
    pipe = build_pipeline(model_config=model_config,
                          usar_smote=usar_smote)

    # Configurar KFolds Estratificados
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # Construir Grid Search con CV
    grid = GridSearchCV(pipe, model_config['param_grid'], cv=cv,
                       scoring='f1_micro', n_jobs=-1, verbose=2,
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
    # T================ TEST ===================== 
    y_test_pred = grid.predict(X_test)

    # Métricas test
    acc_test, f1_test = return_classification_metrics(
        y_test, y_test_pred
    )

    # =================== Classification Report =============
    class_report = classification_report(y_test, y_test_pred)
    os.makedirs(f"{dir_path}/results", exist_ok=True)
    with open(f"{dir_path}/results/{model_name.replace(' ', '_')}_classification_report.txt", "w") as f:
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
        'n_clases': n_clases_str,
        'model_name': model_name,
        'accuracy_train': acc_train,
        'accuracy_test': acc_test,
        'f1_train': f1_train,
        'f1_test': f1_test,
        'best_params': grid.best_params_,
        'class_distribution': class_dist,
        'classification_report': class_report_dict,
        'confusion_matrix_test': cm_test,
        'grid_search': grid,
        'shap_values': shap_values,
        'X_scaled_df': X_scaled_df
    }

    return resultados


def train_test_model_pca(df_imputed, n_clases, model_name, model_config, element = "Nitrogen",
                                usar_smote=False, mostrar_graficos=True, calcular_shap=True,
                                dir_path= "../", n_components=2, CFG=None, best_variables = None):
    """Función principal para entrenar y evaluar un modelo, transformando los datos de entrada con PCA

    Args:
        df_imputed (DataFrame): DataFrame con datos imputados.
        n_clases (int): Número de clases para la codificación.
        model_name (str): Nombre del modelo.
        model_config (dict): Configuración del modelo (estimator y param_grid).
        element (str): Elemento a utilizar para la codificación individual.
        usar_smote (bool): Si se debe usar SMOTE para balancear clases.
        mostrar_graficos (bool): Si se deben mostrar gráficos de confusión.
        calcular_shap (bool): Si se deben calcular valores SHAP.
        dir_path (str): Ruta para almacenar los modelos de cada algoritmo
                        en formato binario .pkl
        CFG: Configuration object
        best_variables (list): Lista de las mejores variables a utilizar. Si es None, se utilizan todas las variables.
    Returns:
        dict: Resultados del entrenamiento y evaluación del modelo.
    """
   # Preparar datos
    X_train_original, X_test_original, y_train, y_test, feature_names, class_dist = preparar_datos(
        df_imputed, n_clases, element=element, best_variables=best_variables, CFG=CFG
    )
    # Aplicar PCA
    X_train, pca = calcuate_PCA(X_train_original, n_components=n_components)
    X_test = pca.transform(X_test_original)
    # Construir objeto de pipeline
    pipe = build_pipeline(model_config=model_config,
                          usar_smote=usar_smote)

    # Configurar KFolds Estratificados
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # Construir Grid Search con CV
    grid = GridSearchCV(pipe, model_config['param_grid'], cv=cv,
                       scoring='f1_micro', n_jobs=-1, verbose=2,
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
    # T================ TEST ===================== 
    y_test_pred = grid.predict(X_test)

    # Métricas test
    acc_test, f1_test = return_classification_metrics(
        y_test, y_test_pred
    )

    # =================== Classification Report =============
    class_report = classification_report(y_test, y_test_pred)
    os.makedirs(f"{dir_path}/results", exist_ok=True)
    with open(f"{dir_path}/results/{model_name.replace(' ', '_')}_classification_report.txt", "w") as f:
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
        shap_values, X_scaled_df = calculate_shap(grid, X_train,
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
        'n_clases': n_clases_str,
        'model_name': model_name,
        'accuracy_train': acc_train,
        'accuracy_test': acc_test,
        'f1_train': f1_train,
        'f1_test': f1_test,
        'best_params': grid.best_params_,
        'class_distribution': class_dist,
        'classification_report': class_report_dict,
        'confusion_matrix_test': cm_test,
        'grid_search': grid,
        'shap_values': shap_values,
        'X_scaled_df': X_scaled_df
    }

    return resultados 

def train_test_model_all_predictions(df_imputed, n_clases, model_name, model_config, element = "Nitrogen",
                              usar_smote=True, mostrar_graficos=True, calcular_shap=True,                              
                              dir_path= "../",
                              best_variables = None,
                              CFG=None
                              ):
    """Función principal para entrenar y evaluar un modelo.
    Separa los datos por KFolds, en cada uno entrena el mismo modelo base con GridSearchCV y se obtiene la predicción de ese fold.
    Al final de cada KFold se obtienen los resultados del la predicción de esos datos y se van acumulando
    Utiliza Crossvalidación en GridSearch.
    Al salir del bucle se tienen todas las predicciones de todos los datos y se calculan las métricas globales.
    En este caso no se puede obtener el modelo final, ya que se entrena un modelo diferente en cada fold, pero se obtiene la predicción de cada fold y se acumula para obtener métricas globales.
    Sin embargo, para SHAP se puede tomar el mejor modelo del último KFold entrenado.
    

    Args:
        df_imputed (DataFrame): DataFrame con datos imputados.
        n_clases (int): Número de clases para la codificación.
        model_name (str): Nombre del modelo.
        model_config (dict): Configuración del modelo (estimator y param_grid).
        element (str): Elemento a utilizar para la codificación individual.
        usar_smote (bool): Si se debe usar SMOTE para balancear clases.
        mostrar_graficos (bool): Si se deben mostrar gráficos de confusión.
        calcular_shap (bool): Si se deben calcular valores SHAP.
        dir_path (str): Ruta para almacenar los modelos de cada algoritmo
                        en formato binario .pkl
        best_variables (list): Lista opcional de variables mas importantes
        CFG: Configuration object
    Returns:
        dict: Resultados del entrenamiento y evaluación del modelo.
    """
    # Extraer datos completos (sin split inicial)
    X, y, feature_names, class_dist = extract_X_y_classification(
        df_imputed, n_clases=n_clases, element=element, CFG=CFG
    )
    
    # Configurar KFold externo
    n_outer_splits = 5
    skf_outer = StratifiedKFold(n_splits=n_outer_splits, shuffle=True, random_state=42)
    skf_inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    # Almacenar predicciones y ground truth de todos los folds
    all_y_true = []
    all_y_pred = []
    all_fold_results = []
    best_model = None
    best_X_train_scaled = None
    best_fold_f1 = -1
    
    # Iterar sobre cada fold externo
    for fold_idx, (train_index, val_index) in enumerate(skf_outer.split(X, y)):

        # Dividir datos
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        
        # Construir pipeline con el mismo modelo
        pipe = build_pipeline(model_config=model_config, usar_smote=usar_smote)
        
        # GridSearchCV con CV interno
        grid = GridSearchCV(
            estimator=pipe,
            param_grid=model_config['param_grid'],
            cv=skf_inner,
            scoring='f1_micro',
            n_jobs=-1,
            verbose=1,
            return_train_score=True
        )
        
        grid.fit(X_train, y_train)
        
        # Predicciones en el fold de validación
        y_pred_val = grid.predict(X_val)
        
        # Acumular predicciones y ground truth
        all_y_true.extend(y_val.tolist())
        all_y_pred.extend(y_pred_val.tolist())
        
        # Calcular métricas del fold
        avg = "weighted" if n_clases is not None else "micro"
        fold_acc, fold_prec, fold_rec, fold_f1 = calculate_classification_metrics(
            y_val, y_pred_val, average=avg
        )
        
        # Guardar resultados del fold
        fold_result = {
            'fold': fold_idx + 1,
            'best_params': grid.best_params_,
            'accuracy': fold_acc,
            'precision': fold_prec,
            'recall': fold_rec,
            'f1_score': fold_f1
        }
        all_fold_results.append(fold_result)
        
        print(f"Fold {fold_idx + 1} - F1: {fold_f1:.4f}, Accuracy: {fold_acc:.4f}")
        
        # Guardar el mejor modelo basado en F1 score
        if fold_f1 > best_fold_f1:
            best_fold_f1 = fold_f1
            best_model = grid.best_estimator_
            scaler = best_model.named_steps['scaler']
    
    # Convertir a arrays numpy
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    
    # Calcular métricas globales sobre todas las predicciones acumuladas
    avg = "weighted" if n_clases is not None else "micro"
    global_acc, global_prec, global_rec, global_f1 = calculate_classification_metrics(
        all_y_true, all_y_pred, average=avg
    )
    
    # Matriz de confusión global
    cm_global = confusion_matrix(all_y_true, all_y_pred)
    
    # Classification Report global
    class_report = classification_report(all_y_true, all_y_pred)
    print(f"\n{class_report}")
    
    class_report_dict = classification_report(all_y_true, all_y_pred, output_dict=True)
    
    # Guardar classification report
    os.makedirs(f"{dir_path}/results", exist_ok=True)
    if CFG.individual_train:
        report_filename = f"{dir_path}/results/{model_name.replace(' ', '_')}_nclases_3_{element}_all_predictions_report.txt"
    elif CFG.cuartiles_train:
        report_filename = f"{dir_path}/results/{model_name.replace(' ', '_')}_nclases_2_cuartiles_all_predictions_report.txt"
    else:
        report_filename = f"{dir_path}/results/{model_name.replace(' ', '_')}_nclases_{n_clases}_all_predictions_report.txt"
    
    with open(report_filename, "w") as f:
        f.write(class_report)
    
    # Gráfico de confusión global
    fig_cm_global = plot_confusion_matrix(
        cm_global, 
        classes=np.unique(all_y_true),
        title=f"{model_name} - Global Confusion Matrix ({n_outer_splits} folds)"
    )
    if mostrar_graficos:
        plt.show()
    
    # Guardar el mejor modelo
    os.makedirs(f"{dir_path}/models", exist_ok=True)
    if best_model is not None:
        if CFG.individual_train:
            model_path = f"{dir_path}/models/{model_name.replace(' ', '_')}_nclases_3_{element}_all_predictions.pkl"
        elif CFG.cuartiles_train:
            model_path = f"{dir_path}/models/{model_name.replace(' ', '_')}_nclases_2_cuartiles_all_predictions.pkl"
        else:
            model_path = f"{dir_path}/models/{model_name.replace(' ', '_')}_nclases_{n_clases}_all_predictions.pkl"
        
        with open(model_path, 'wb') as f:
            pickle.dump(best_model, f)
        print(f"\nBest model saved: {model_path}")
    
    # ================ Calcular SHAP ===========================
    shap_values = None
    X_scaled_df = None
    if calcular_shap and best_model is not None:
        # Para SHAP, entrenar el modelo con todos los datos usando los mejores parámetros del mejor fold
        
        X_train, X_test, y_train, y_test, feature_names, class_dist = preparar_datos(
            df_imputed, n_clases=n_clases, element=element, best_variables=best_variables, CFG=CFG
        )
        
        # Extraer los mejores parámetros del mejor modelo (del fold con mejor F1)
        # best_model es un Pipeline con: scaler -> (smote) -> clf
        best_clf_params = best_model.named_steps['clf'].get_params()
        
        # Crear un nuevo pipeline con el mismo modelo base pero con los mejores parámetros
        final_pipe = build_pipeline(model_config=model_config, usar_smote=usar_smote)
        
        # Establecer los mejores parámetros encontrados en el clasificador del pipeline
        final_pipe.named_steps['clf'].set_params(**best_clf_params)
        
        # Entrenar con todos los datos usando los mejores parámetros
        final_pipe.fit(X_train, y_train)
        
        # Preparar datos para SHAP
        scaler = final_pipe.named_steps['scaler']
        best_X_test_scaled = scaler.transform(X_test)
        
        shap_values, X_scaled_df = calculate_shap(
            final_pipe, best_X_test_scaled,
            dir_path, model_name,
            feature_names, model_config,
            n_clases,
            mostrar_graficos
        )
    
    # ============ Almacenar Resultados ===========================
    if CFG.individual_train:
        n_clases_str = f"{3}_{element}"
    elif CFG.cuartiles_train:
        n_clases_str = "2_Quartiles"
    else:
        n_clases_str = str(n_clases)
    
    resultados = {
        'n_clases': n_clases_str,
        'model_name': model_name,
        'accuracy_train': global_acc,  # Solo hay predicciones de validación
        'accuracy_test': global_acc,
        'precision_train': global_prec,
        'precision_test': global_prec,
        'recall_train': global_rec,
        'recall_test': global_rec,
        'f1_train': global_f1,
        'f1_test': global_f1,
        'class_distribution': class_dist,
        'classification_report': class_report_dict,
        'confusion_matrix_test': cm_global,
        'fold_results': all_fold_results,
        'all_predictions': {'y_true': all_y_true, 'y_pred': all_y_pred},
        'shap_values': shap_values,
        'X_scaled_df': X_scaled_df,
        'best_params': best_model.named_steps['clf'].get_params() if best_model else None,
        'best_model': best_model
    }

    return resultados

def compare_classification_models(resultados_dict, CFG):
    """Compara el rendimiento de diferentes modelos de machine Learning y sus respectivos modelos
    según la codificación realizada. Por ejemplo Modelos N, P, K, o modelo de cuartiles de 
    productividad .
    Realiza la grafica de barras de rendimiento de los resultados.

    Args:
        resultados_dict (dict):
            Diccionario con resultados de cada modelo
            Formato: {model_name: [lista de resultados por n_clases]}
    """
    def plot_comparative_bars(resultados_dict: dict, metric: str):

        if CFG.individual_train:
            n_classes = CFG.elements_list
        elif CFG.cuartiles_train:
            n_classes = ["Quartiles"]
        else:
            n_classes = np.arange(2, 10)

        fig, ax = plt.subplots(figsize=(10, 6))
        width = 0.15  # the width of the bars
        n_models = len(resultados_dict)
        multiplier = 0
        x = np.arange(len(n_classes))
        for model_name, resultados in resultados_dict.items():
            metrics = [resultados[i][metric] for i in range(len(resultados))]

            # Center the bars by offsetting from the middle
            offset = width * (multiplier - n_models / 2 + 0.5)
            rects = ax.bar(x + offset, metrics, width, label=model_name, 
                          edgecolor='black', linewidth=0.8)
            #ax.bar_label(rects, padding=8,  fmt='{:.3f}', rotation = 45)

            # Poner Etiquetas
            for rect in rects:
                height = rect.get_height()
                txt = ax.text(rect.get_x() + rect.get_width()/2,
                        height/2,  # Posición en el centro de la barra
                        f'{height:.3f}',
                        ha='center',  # alineación horizontal centrada
                        va='center',  # alineación vertical centrada
                        rotation=90,
                        fontsize=10,
                        fontweight='bold',
                        color='white')
                txt.set_path_effects([path_effects.Stroke(linewidth=2, foreground='black'),
                                     path_effects.Normal()])

            multiplier += 1

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
        ax.set_title(f'Comparison of Models by {metric.replace("_", " ").title()}', fontsize=14, pad=15)
        ax.set_xticks(x, n_classes)
        if CFG.individual_train:
            ax.set_xlabel('Fertilizer Element', fontsize=12)
        elif CFG.cuartiles_train:
            ax.set_xlabel('Model', fontsize=12)
        else:
            ax.set_xlabel('Number of Classes', fontsize=12)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_ylim(0, 1)
        # Save figure
        plt.tight_layout()
        plt.savefig(f'{CFG.class_path}comparison_{metric}.png', dpi=300, bbox_inches='tight')
        plt.show()

    # Gráficos comparativos
    plot_comparative_bars(resultados_dict, 'accuracy_test')
    plot_comparative_bars(resultados_dict, 'f1_test')
    # Tabla resumen
    print("\n" + "="*80)
    print("SUMMARY OF BEST RESULTS BY MODEL")
    print("="*80)
    for model_name, resultados in resultados_dict.items():
        df_temp = pd.DataFrame(resultados)
        best_idx = df_temp['f1_test'].idxmax()
        best_result = df_temp.iloc[best_idx]
        print(f"\n{model_name}:")
        if CFG.individual_train:
            print(f"  Best element: {best_result['n_clases']}")
        else:
            print(f"  Best n_classes: {best_result['n_clases']}")
        print(f"  Accuracy: {best_result['accuracy_test']:.4f}")
        print(f"  F1 Macro: {best_result['f1_test']:.4f}")

def save_results_general(all_results:dict, dir_class):
    resultados_completos = []

    for model_name, resultados in all_results.items():
        for resultado in resultados:
            resultados_completos.append({
                'Model': model_name,
                'N_Classes': resultado['n_clases'],
                'Accuracy_Train': resultado['accuracy_train'],
                'Accuracy_Test': resultado['accuracy_test'],
                'F1_Train': resultado['f1_train'],
                'F1_Test': resultado['f1_test'],
                'Best_Params': str(resultado['best_params'])
            })

    df_resultados = pd.DataFrame(resultados_completos)

    # Guardar resultados
    df_resultados.to_csv(f'{dir_class}resultados_modelos_completos.csv', index=False)

def save_model(model, path):
    with open(path, 'wb') as f:
        pickle.dump(model, f)
def load_pickle_results(path):
    with open(path, 'rb') as f:
        loaded_model = pickle.load(f)
    return loaded_model

# =====================================================================
# ======= SHAP VALUES ANALYSIS ======================================
# ==================================================================


def global_shap_N_class(results_model_i, model_name, dir_path = ".", CFG=None) -> pd.DataFrame:
    """Calcula la importancia gloal de SHAP dada una lista de valores shap para una clase específico.
    Se busca hacer promedio absoluto de las importancias SHAP por clase.
    Args:
        results_model_i (list): Lista de resultados del modelo. Internamente se espera que cada elemento tenga 'shap_values' y 'X_scaled_df'.
                                'shap_values' debe tener forma (n_samples, n_features, n_classes) o (n_samples, n_features) para binario.
        model_name (str): Nombre del modelo.
    """
    global_variables =  np.zeros(results_model_i[0]['X_scaled_df'].shape[1]) # se espera que sea 26 o 20 
    global_df = pd.DataFrame()
    for i in range(len(results_model_i)):
        #print(f"\nTop variables for n_classes = {i+2}:")
        shap_values = results_model_i[i]['shap_values']
        X_scaled_df = results_model_i[i]['X_scaled_df']
        
        # Handle both 2D (binary) and 3D (multi-class) SHAP arrays
        if len(shap_values.shape) == 2:
            # Binary classification: (n_samples, n_features)
            shap_importance = np.mean(np.abs(shap_values), axis=0)
        else:
            # Multi-class: (n_samples, n_features, n_classes)
            shap_importance = np.mean(np.abs(shap_values), axis=(0, 2))
        
        # sumar valores de shap importance
        global_variables += shap_importance
        if CFG.individual_train or CFG.cuartiles_train:
            global_df_temp = pd.DataFrame({
                f'{results_model_i[i]["n_clases"]}': shap_importance
            })
        else:
            global_df_temp = pd.DataFrame({
                f'class_{i+2}': shap_importance
            })
        global_df = pd.concat([global_df, global_df_temp], axis=1)
        #print(shap_importance)
    # poner columna de features en la primera columna
    global_df.insert(0, 'feature', results_model_i[0]['X_scaled_df'].columns)

    model_name = model_name.replace(" ", "_")
    #dir_path = f"../Resultados/classification/{model_name}"

    #save csv
    #global_df.to_csv(f"{dir_path}/global_shap_importance_by_class.csv", index=False)

    top_global_variables = pd.DataFrame({
        'feature': results_model_i[0]['X_scaled_df'].columns,
        'shap_importance': global_variables / (len(results_model_i))
    })
    top_global_variables = top_global_variables.sort_values(by='shap_importance', ascending=False).reset_index(drop=True)
    #top_global_variables.to_csv(f"{dir_path}/top_global_variables_mean.csv", index=False)
    return global_df

def assign_ranking_weights(df, weight_method='percentage', CFG=None):
    """Asigna ponderaciones según el ranking de importancia SHAP.

    Args:
        df (pd.DataFrame): DataFrame con columnas 'feature' y 'class_{i}'.
        weight_method (str): Método de ponderación:
            - 'linear': 1/posición (ej: 1, 1/2, 1/3, ...)
            - 'exponential': 2^(n-posición) donde n es el total de features
            - 'inverse_square': 1/posición^2
            - 'rank': n - posición + 1 (ej: n, n-1, n-2, ...)
    Returns:
        pd.DataFrame: DataFrame con ponderaciones por clase.
    """
    n_features = len(df)
    if CFG.individual_train:
        class_columns = [col for col in df.columns if col.startswith('3')]
    elif CFG.cuartiles_train:
        class_columns = [col for col in df.columns if col.startswith('2')]
    else:
        class_columns = [col for col in df.columns if col.startswith('class_')]

    # DataFrame para almacenar las ponderaciones
    weights_df = pd.DataFrame()
    weights_df['feature'] = df['feature']
    # display(df)
    for class_col in class_columns:
        # Ordenar por importancia SHAP descendente
        sorted_df = df[['feature', class_col]].sort_values(by=class_col, ascending=False).reset_index(drop=True)

        # Calcular ponderaciones según el método elegido
        if weight_method == 'linear':
            # 1/posición: 1, 1/2, 1/3, ..., 1/n
            weights = 1 / (sorted_df.index + 1)
        elif weight_method == 'exponential':
            # 2^(n-posición): más peso a los primeros
            weights = 2 ** (n_features - sorted_df.index - 1)
        elif weight_method == 'inverse_square':
            # 1/posición^2: decae más rápido
            weights = 1 / ((sorted_df.index + 1) ** 2)
        elif weight_method == 'rank':
            # n - posición + 1: n, n-1, n-2, ..., 1
            weights = n_features - sorted_df.index
        elif weight_method == 'percentage':
            # Obtener porcentage de cada valor shap
            weights = sorted_df[class_col] / sorted_df[class_col].sum()
        else:
            raise ValueError(f"Método '{weight_method}' no reconocido")

        # Crear DataFrame temporal con ponderaciones
        temp_df = pd.DataFrame({
            'feature': sorted_df['feature'],
            f'weight_{class_col}': weights
        })

        # Merge para mantener el orden original de features
        weights_df = weights_df.merge(temp_df, on='feature', how='left')

    return weights_df

def average_models_by_order(results_models_all, dir_path_base, weight_method):
    """
    
    """
    df_sum = None
    for model_name, results_model_i in results_models_all.items():
        dir_path = f"{dir_path_base}/{model_name.replace(' ', '_')}"
        #save shap values for all classes
        global_shap_model_i =  global_shap_N_class(results_model_i, model_name, dir_path)

        # Assign values according the its ranking
        weighted_rf_model_i = assign_ranking_weights(global_shap_model_i, weight_method=weight_method)

        cols = weighted_rf_model_i.columns[1:] # don't include feature column
        if df_sum is None:
            df_sum = weighted_rf_model_i.copy()
        else:
          df_sum[cols] += weighted_rf_model_i[cols] # Acumulate according to the ranking

    df_sum[cols] = df_sum[cols] / len(results_models_all) # Averagin according to the number of models
    return df_sum

def top_variables_by_algorithm(results_models_all, dir_path_base):
    for model_name, results_model_i in results_models_all.items():
        dir_path = f"{dir_path_base}/{model_name.replace(' ', '_')}"
        #save shap values for all classes
        global_shap_model_i =  global_shap_N_class(results_model_i, model_name, dir_path)

        
        for col in global_shap_model_i.columns:
            if col.startswith('class_'):
                # Sort global_shap_model_i by the current class column in descending order
                sorted_features = global_shap_model_i.sort_values(by=col, ascending=False)['feature'].reset_index(drop=True)
                # Add the sorted features as a new column to top_features_by_class DataFrame
                col_idx = global_shap_model_i.columns.get_loc(col)
                # Insert the sorted features right after the current column
                global_shap_model_i.insert(col_idx + 1, f'Feature {col}', sorted_features)
            
        global_shap_model_i.rename(columns={'feature': f'feature - {model_name}'}, inplace=True)
        sub_dir_path = f"{dir_path_base}/top_variables_by_algorithm"
        os.makedirs(sub_dir_path, exist_ok=True)
        global_shap_model_i.to_csv(f"{sub_dir_path}/top_variables_by_model_class_{model_name.replace(' ', '_')}.csv", index=False)
        #display(global_shap_model_i)


def extract_top_x_percent_features(results_models_all, percent=0.8, class_path = ".",
                                    CFG=None):
    """
    Extrae las features que acumulan el porcentaje especificado de importancia SHAP para cada modelo y clase.
    Args:
        results_models_all (dict): Diccionario con los resultados de todos los modelos.
        percent (float): Porcentaje acumulado de importancia SHAP para seleccionar las features (ej: 0.8 para el 80%).
        class_path (str): Ruta del directorio donde se guardarán los archivos CSV con las features seleccionadas.
    Returns:
        dict: Diccionario con las features seleccionadas para cada modelo y clase.
              Formato: {Category: {model_name: [features]}}
    """
    # Variable para guardar las features (será la misma para todos)
    features = None
    best_x_percentage_all_algorithms = {}

    for model_name, results_model_i in results_models_all.items():
        dir_path = f"{class_path}/{model_name.replace(' ', '_')}"
        # Save shap values for each algorithm
        global_shap_model_i =   global_shap_N_class(results_model_i, model_name, dir_path, CFG=CFG)
        ranked_df = assign_ranking_weights(global_shap_model_i, weight_method='percentage', CFG=CFG)

        for col in ranked_df.columns:
            if col.startswith('weight_'):
                # Obtener el 80% de las features
                sorted_df = ranked_df[['feature', col]].sort_values(by=col, ascending=False).reset_index(drop=True)
                sorted_df['acumulated'] = sorted_df[col].cumsum() / sorted_df[col].sum()
                #display(sorted_df)
                # Recorrer todas las features y parar hasta que el acumulado sea el 80%
                cumulative_sum = 0
                top_features = []
                for index, row in sorted_df.iterrows():
                    cumulative_sum += row[col]
                    top_features.append(row['feature'])
                    if cumulative_sum >= percent:
                        break
                
                # Inicializar la columna si no existe
                if col not in best_x_percentage_all_algorithms:
                    best_x_percentage_all_algorithms[col] = {}
                
                # Guardar las top features para este modelo y columna
                best_x_percentage_all_algorithms[col][f"{col}_{model_name}"] = top_features
                print(f"  {len(top_features)} features que acumulan {cumulative_sum:.2%}")

    # almacenar en csv
    for col, models_dict in best_x_percentage_all_algorithms.items():
        df_temp = pd.DataFrame()
        for model, features_list in models_dict.items():
            df_temp[model] = pd.Series(features_list)
        df_temp.to_csv(f"{class_path}/best_{int(percent*100)}_percent_features_{col}.csv", index=False)
    return best_x_percentage_all_algorithms



# =====================================================================
# ======= SHAP VALUES ANALYSIS ======================================
# ==================================================================



def category_shap_values(results_model_i):
    """
    Extrae los valores SHAP promedio absoluto por categoria para un modelo específico.
    e.g. Las categorías del modelo de 3 clases son: clase 0, clase 1, clase 2 (deficiencia, adecuado, exceso)
    Para dos clases sería: clase 0, clase 1 (tratamientos en q1, tratamientos en q2)

    Args:
        results_model_i (list): Lista de resultados de un algoritmo específico.
                                Cada elemento es para un modelo con n_clases específico.
                                Debe incluir la matriz 'shap_values' y 'n_clases'.
    Returns:
        dict: Diccionario con los valores SHAP promedio absoluto por categoria.
            e.g {3_Nitrogen: [[],[],[]], 3_Phosphorus: [[],[],[]], 3_Potassium: [[],[],[]]}
    """
    shap_by_class = {}
    # Iterar la lista de reusltados( cada elemento es un modelo o nclase)
    for i in range(len(results_model_i)): 
        shap_values = results_model_i[i]['shap_values']
        shap_mean_per_class = np.mean(np.abs(shap_values), axis=0)  # Promedio absoluto por clase        
        shap_by_class[results_model_i[i]['n_clases']] = shap_mean_per_class
    
    return shap_by_class

def category_algorithm_shap_values(results_models_all):
    """
    Calcula los valores SHAP promedio absoluto por categoria y por algoritmo.
    Args:
        results_models_all (dict): Diccionario con los resultados de todos los modelos.
                                    Formato: {algorithm_name: [lista de resultados por n_clases]}
    Returns:
        dict: Diccionario con los valores SHAP por clase y algoritmo.
              Formato: {model_name: {algorithm_name: shap_values}}
    """
    print("Extrayendo valores SHAP por clase y algoritmo...")
    shap_values_all = {}
    for alg_name, results_model_i in results_models_all.items():
        shap_values_all[alg_name] = category_shap_values(results_model_i)
    shap_value_by_class = {}
    #invertir asociación: las claves ahora son las n_classes, y por dentro va cada algoritmo con shap values correspondientes
    model_names = list(shap_values_all[alg_name].keys())
    for n_class in model_names:
        shap_value_by_class[n_class] = {}
        for alg_name in shap_values_all.keys():
            shap_value_by_class[n_class][alg_name] = shap_values_all[alg_name][n_class]
    return shap_value_by_class

def save_shap_category_algorithm_csv(dict_class_algrtm_shap, 
                                      results_models_all, 
                                      dir_path, CFG):
    """
    Almacena los valores SHAP por categoria y algoritmo en archivos CSV.
    Args:
        dict_class_algrtm_shap (dict): Diccionario con los valores SHAP por clase y algoritmo.
                                       Formato: {n_class: {algorithm_name: shap_values}}
        results_models_all (dict): Diccionario con los resultados de todos los modelos.
        dir_path (str): Ruta del directorio donde se guardarán los archivos CSV.
    Returns:
        None
    """
    print("Guardando valores SHAP por categoria y algoritmo...")
    for n_class, dict_alg_shap in dict_class_algrtm_shap.items():
        df_temp = pd.DataFrame()
        for alg_name, shap_values in dict_alg_shap.items():
            # si el modelo es XGB y es binario (cuartiles train)
            # crear una columna por cada categoria
            if alg_name == "XGB" and CFG.cuartiles_train:
                # duplicar la columna por compatibilidad
                shap_values = np.column_stack([shap_values, shap_values])
            #insertar cada columna de shap_values corresponde a una categoria diferente.
            for class_idx in range(shap_values.shape[1]):
                df_temp[f'{alg_name}_category_{class_idx}'] = shap_values[:, class_idx]
        df_temp.insert(0, 'feature', results_models_all[list(results_models_all.keys())[0]][0]['X_scaled_df'].columns)
        df_temp.to_csv(f"{dir_path}/shap_values_category_{n_class}.csv", index=False)

    
def variable_importance_category_algorithm_shap(dict_class_algrtm_shap,
                                                percentage,
                                                results_models_all, dir_path, CFG):
    """
    Ordena las variables más importantes según los valores SHAP por categoria y algoritmo.
    Se filtran las variables hasta que el porcentaje acumulado de importancia SHAP
    sea igual al porcentaje dado.

    Args:
        dict_class_algrtm_shap (dict): Diccionario con los valores SHAP por clase y algoritmo.
                                       Formato: {model_name: {algorithm_name: shap_values}}
        percentage (float): Porcentaje acumulado de importancia SHAP para filtrar variables.
        results_models_all (dict): Diccionario con los resultados de todos los modelos.
        dir_path (str): Ruta del directorio donde se guardarán los archivos CSV.
    Returns:
        dict: Diccionario con los rankings de variables por clase y en cada algoritmo.
    """
    
    ranking_vars_class = {}
    for n_class, dict_alg_shap in dict_class_algrtm_shap.items():
        ranking_vars_class[n_class] = {}
        for alg_name, shap_values in dict_alg_shap.items():
            # si el modelo es XGB y es binario (cuartiles train)
            # crear una columna por cada categoria
            if alg_name == "XGB" and CFG.cuartiles_train:
                # duplicar la columna por compatibilidad
                shap_values = np.column_stack([shap_values, shap_values])
            for class_idx in range(shap_values.shape[1]):
                # Obtener el ranking de variables para esta clase
                shap_importance = shap_values[:, class_idx]
                #pasar a porcentaje
                shap_importance_p = shap_importance / np.sum(shap_importance)
                ranked_indices = np.argsort(-shap_importance_p)  # Orden descendente

                print(results_models_all[list(results_models_all.keys())[0]][0]['X_scaled_df'].columns[ranked_indices])
                # Filtrar por porcentaje acumulado
                cumulative_importance = np.cumsum(shap_importance_p[ranked_indices])
                num_features = np.sum(cumulative_importance <= percentage) + 1  # +1 para incluir la característica que cruza el umbral
                ranked_indices = ranked_indices[:num_features]

                # Asignar nombres correspondientes de variables
                ranked_features = results_models_all[list(results_models_all.keys())[0]][0]['X_scaled_df'].columns[ranked_indices]

                if f'class_{class_idx}' not in ranking_vars_class[n_class]:
                    ranking_vars_class[n_class][f'class_{class_idx}'] = {}
                ranking_vars_class[n_class][f'class_{class_idx}'][alg_name] = ranked_features.tolist()
    # crear carpeta si no existe
    name_folder = os.path.join(dir_path, "ranking_variables_by_class_algorithm")
    os.makedirs(name_folder, exist_ok=True)
    # ====================guardar en csv, por clase y por nombre de modelo ====================
    print(f"Guardando rankings de variables por clase y algoritmo en {name_folder}") 
    for n_class, dict_alg_rankings in ranking_vars_class.items():
        for class_name, dict_model_rankings in dict_alg_rankings.items():
            df_temp = pd.DataFrame()
            for alg_name, ranked_features in dict_model_rankings.items():
                df_temp[alg_name] = pd.Series(ranked_features)
            df_temp.to_csv(f"{name_folder}/ranking_vars_nclass_{n_class}_{class_name}_{int(percentage*100)}.csv", index=False)
    return ranking_vars_class

def analyze_variable_df(df):
    """
    Analiza las variables más frecuentes en el top N de diferentes modelos.
    
    Args:
        df (pd.DataFrame): DataFrame donde cada columna representa un modelo/clase
                           y cada fila representa una variable en orden de importancia.
            Formato esperado:
            Model1, Model2, Model3, ...
            var1,   var2,   var3, ...
            var4,   var5,   var6, ...
    
    Returns:
        tuple: (DataFrame con frecuencias, DataFrame con posiciones y frecuencias)
    
    """
     # Diccionario para almacenar la frecuencia de cada variable
    variable_frequency = {}
    # Diccionario para almacenar las posiciones de cada variable
    variable_positions = {}
    
    # Iterar sobre cada columna (cada modelo/clase)
    for col in df.columns:
        # Obtener las top N variables de esta columna
        top_vars = df[col].tolist()
        
        # Contar frecuencia y registrar posiciones
        for position, var in enumerate(top_vars, start=1):
            if var not in variable_frequency:
                variable_frequency[var] = 0
                variable_positions[var] = []
            
            variable_frequency[var] += 1
            variable_positions[var].append({
                'model_class': col,
                'position': position
            })
    
    # Crear DataFrame con frecuencias
    freq_df = pd.DataFrame(list(variable_frequency.items()), 
                          columns=['Variable', 'Frequency'])
    freq_df = freq_df.sort_values('Frequency', ascending=False).reset_index(drop=True)
    # Crear DataFrame con posiciones detalladas
    position_data = []
    for var, positions in variable_positions.items():
        for pos_info in positions:
            position_data.append({
                'Variable': var,
                'Model_Class': pos_info['model_class'],
                'Position': pos_info['position'],
                'Frequency': variable_frequency[var]
            })
    
    position_df = pd.DataFrame(position_data)
    
    return freq_df, position_df

def save_top_variable_by_category(ranking_vars_class, percentage, dir_path):
    """
    Guarda las mejores variables por categoría de cada modelo, ponderando
    las variables presentes en todos los algoritmos. Se toma hasta que el porcentaje
    acumulado de importancia SHAP sea igual al porcentaje dado.

    Args:
        ranking_vars_class (dict): Diccionario con los rankings de variables por clase y en 
                                    cada algoritmo.
        percentage (float): Porcentaje acumulado de importancia SHAP para filtrar variables.
        dir_path (str): Ruta del directorio donde se guardarán los archivos CSV.
    Returns:
        None
    """
    print("Guardando top variables por clase y algoritmo...")
    name_folder = os.path.join(dir_path, "ranking_variables_by_class_algorithm")
    os.makedirs(name_folder, exist_ok=True)
    for n_class, dict_alg_rankings in ranking_vars_class.items():
        best_variables_by_class_alg = {}
        for class_name, dict_model_rankings in dict_alg_rankings.items():
            df_temp = pd.DataFrame()
            for alg_name, ranked_features in dict_model_rankings.items():
                df_temp[alg_name] = pd.Series(ranked_features)
            print(f"{class_name} - n_class {n_class}")
            #display(df_temp)
            top_vars_freq, top_vars_positions = analyze_variable_df(df_temp)

            #-------------------------------------
            ## Sumar posiciones para cada variable
            position_summary = top_vars_positions.groupby('Variable', as_index=False).agg({
                'Position': 'sum',
                'Frequency': 'first'  # o 'max' ya que todos tienen el mismo valor
            }).rename(columns={'Position': 'Position_Sum'})

            # Ordenar por Frequency y Position_Sum
            position_summary = position_summary.sort_values(
                        by=['Frequency', 'Position_Sum'], 
                        ascending=[False, True]  
                        ).reset_index(drop=True)
            #display(position_summary)
            #display(analyze_variable_df(df_temp))#.to_csv(f"{name_folder}/variable_frequency_nclass_{n_class}_{class_name}.csv", index=False)
            best_variables_by_class_alg[class_name] = position_summary['Variable'].tolist()
        # guardar mejores variables por clase en todos los algoritmos
        df_best_vars = pd.DataFrame()
        for class_name, var_list in best_variables_by_class_alg.items():
            df_best_vars[class_name] = pd.Series(var_list)
        df_best_vars.to_csv(f"{name_folder}/ponderated_{int(percentage*100)}_best_vars_nclass_{n_class}_all_algorithms.csv", index=False)

# ====================================================================
# VARIABLE FRECUENCY ANALYSIS
# ====================================================================

def analyze_top_variables(csv_path):
    """
    Analiza las variables más frecuentes en el top N de diferentes modelos.
    
    Args:
        csv_path (str): Ruta al archivo CSV con las variables ordenadas por importancia.
        El formato del csv debe ser del tipo:
            Model1, Model2, Model3, ...
            var1,   var2,   var3, ...
            var4,   var5,   var6, ...
            ...
    
    Returns:
        tuple: (DataFrame con frecuencias, DataFrame con posiciones)
    """
    # Leer el CSV
    df = pd.read_csv(csv_path)
    
    # Diccionario para almacenar la frecuencia de cada variable
    variable_frequency = {}
    # Diccionario para almacenar las posiciones de cada variable
    variable_positions = {}
    
    # Iterar sobre cada columna (cada modelo/clase)
    for col in df.columns:
        # Obtener las top N variables de esta columna
        top_vars = df[col].tolist()
        
        # Contar frecuencia y registrar posiciones
        for position, var in enumerate(top_vars, start=1):
            if var not in variable_frequency:
                variable_frequency[var] = 0
                variable_positions[var] = []
            
            variable_frequency[var] += 1
            variable_positions[var].append({
                'model_class': col,
                'position': position
            })
    
    # Crear DataFrame con frecuencias
    freq_df = pd.DataFrame(list(variable_frequency.items()), 
                          columns=['Variable', 'Frequency'])
    freq_df = freq_df.sort_values('Frequency', ascending=False).reset_index(drop=True)
    
    # Crear DataFrame con posiciones detalladas
    position_data = []
    for var, positions in variable_positions.items():
        for pos_info in positions:
            position_data.append({
                'Variable': var,
                'Model_Class': pos_info['model_class'],
                'Position': pos_info['position'],
                'Frequency': variable_frequency[var]
            })
    
    position_df = pd.DataFrame(position_data)
    
    return freq_df, position_df


def plot_top_variables_analysis(freq_df, element=None, percentage=80, dir_path=None):
    """
    Genera gráficos para visualizar el análisis de variables top.
    
    Args:
        freq_df (pd.DataFrame): DataFrame con frecuencias de variables.
        position_df (pd.DataFrame): DataFrame con posiciones detalladas.
        top_variables (int): Número de variables principales a visualizar.
    """
    
    # Crear figura con subplots
    #crear copia para no modificar el original
    freq_df_copy = freq_df.copy()
    # Cambiar "_" por " " en nombres de variables para mejor visualización
    freq_df_copy['Variable'] = freq_df_copy['Variable'].str.replace('_', ' ')
    freq_df_copy['Variable'] = freq_df_copy['Variable'].str.replace('[', ' [')
    #Gráfico de barras: Frecuencia de aparición en top
    
    sns.barplot(data=freq_df_copy, 
                x='Frequency', y='Variable', hue='Variable', palette='viridis', legend=False)
    
    plt.title(f'Variables Most Frequent - {element} Model', fontsize=14, fontweight='bold')
    plt.xlabel(f'Frequency (times appearing in {percentage}%)', fontsize=12)
    plt.ylabel('Variable', fontsize=12)
    plt.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(dir_path + f'top_vars_freq_{percentage}_{element}.png', dpi=300)
    plt.show()

def most_frequent_variables_analysis(csv_path, element=None, percentage=80, dir_path=None):
    """
    Realiza el análisis de las variables más frecuentes de diferentes modelos.
    
    Args:
        csv_path (str): Ruta al archivo CSV con las variables ordenadas por importancia.
        element (str): Nombre del elemento (opcional).
        percentage (int): Porcentaje para definir (por defecto 80).
        dir_path (str): Ruta del directorio para guardar los gráficos (opcional).
    Returns:
        list: Lista de variables que aparecen en todos los modelos.
    """

    top_vars_freq, top_vars_positions = analyze_top_variables(csv_path)

    #-------------------------------------
    ## Sumar posiciones para cada variable
    position_summary = top_vars_positions.groupby('Variable', as_index=False).agg({
        'Position': 'sum',
        'Frequency': 'first'  # o 'max' ya que todos tienen el mismo valor
    }).rename(columns={'Position': 'Position_Sum'})

    # Ordenar por Frequency y Position_Sum
    position_summary = position_summary.sort_values(
                    by=['Frequency', 'Position_Sum'], 
                    ascending=[False, True]  
                    ).reset_index(drop=True)
    #display(position_summary)
    #-------------------------------------
    #display(top_vars_freq)

    # Generar gráfico de barras de las variables más frecuentes
    plot_top_variables_analysis(position_summary, 
                                              element=element,
                                              percentage=percentage,
                                              dir_path=dir_path)
    all_models_count = len(pd.read_csv(csv_path).columns)
    return position_summary[position_summary['Frequency'] == all_models_count]['Variable'].tolist()

# =====================================================================
# ======= PERMUTATION IMPORTANCE ANALYSIS ============================
# =====================================================================

def permutation_importance_all_elements(class_results, df_imputed, dir_path_permutation, list_elements, CFG, all_models = False, best_variables = None):
    
    algorithms = class_results.keys()
    dict_perm_impt_vals = {}
    for i in range(len(list_elements)):
        df = pd.DataFrame()
        dict_perm_impt_vals[list_elements[i]] = {}
        for alg in algorithms:
            X_train, X_test, y_train, y_test, feature_names, class_dist = preparar_datos(
                df_imputed, n_clases=3, element=list_elements[i], CFG=CFG, best_variables=best_variables
            )
            if all_models:
                model = class_results[alg][i]['best_model']
            else:
                # Check if nested CV (dict) or regular GridSearchCV
                if isinstance(class_results[alg][i]['grid_search'], dict):
                    # Nested model: load the saved pipeline from disk
                    if CFG.cuartiles_train or list_elements[i] == 'Quartiles':
                        model_path = f"{CFG.class_path}{alg.replace(' ', '_')}/models/{alg.replace(' ', '_')}_nclases_2_cuartiles.pkl"
                    else:
                        model_path = f"{CFG.class_path}{alg.replace(' ', '_')}/models/{alg.replace(' ', '_')}_nclases_3_{list_elements[i]}.pkl"
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)
                else:
                    # Regular model: use best_estimator_
                    model = class_results[alg][i]['grid_search'].best_estimator_
            perm = PermutationImportance(model, random_state=0, n_iter=15, cv="prefit").fit(X_test, y_test)
            
            print(f'Permutation Importance for {alg} - {list_elements[i]}:')
            
            perm_importances = perm.feature_importances_
            dict_perm_impt_vals[list_elements[i]][alg] = perm_importances
            #agregar a df
            df[alg] = perm_importances
        #agregar nombres de features en primera columna
        df.insert(0, 'Feature', X_test.columns.tolist())
        #display(df)
        #guardar
        df.to_csv(f'{dir_path_permutation}permutation_importance_{list_elements[i]}.csv', index=False)
    return dict_perm_impt_vals


def most_frequent_variables_analysis_PERMUTATION(df_imputed, dict_perm_impt_vals, dir_path_permutation, CFG, best_variables = None):
    percentages = [0.7, 0.8]
    for percentage in percentages:
        for element in dict_perm_impt_vals: #recorre cada modelo (elementos o cuartil)
            best_feats_element = {}
            for alg in dict_perm_impt_vals[element]:
                vals = dict_perm_impt_vals[element][alg]
                vals[vals < 0] = 0
                vals = vals / np.sum(vals)
                X_train, X_test, y_train, y_test, feature_names, class_dist = preparar_datos(
                df_imputed, n_clases=3, element=element, CFG=CFG, best_variables=best_variables
                )
                
                # Revisar porcentaje acumulado
                df = pd.DataFrame({'Variable': X_test.columns.tolist(), 'Importance': vals})
                df = df.sort_values(by='Importance', ascending=False)
                df['Cumulative_Importance'] = df['Importance'].cumsum()

                top_feats = df[df['Cumulative_Importance'] <= percentage]
                #guardar el nombre de las variables en diccionario
                best_feats_element[alg] = top_feats['Variable'].tolist()
            
            # guardar csv
            df_best_feats = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in best_feats_element.items()]))
            print(f"{element}")
            #display(df_best_feats)
            df_best_feats.to_csv(f'{dir_path_permutation}best_{int(percentage*100)}_percent_features_{element}.csv', index=False)

def permutation_importance_NPK(all_results, df_imputed, dir_path_permutation, class_path, CFG, all_models = False):
    
    os.makedirs(dir_path_permutation, exist_ok=True)
    dict_perm_impt_vals = permutation_importance_all_elements(all_results, df_imputed, dir_path_permutation,
                                                            list_elements = ['Nitrogen', 'Phosphorus', 'Potassium'], CFG=CFG, all_models=all_models)
    
    most_frequent_variables_analysis_PERMUTATION(df_imputed, dict_perm_impt_vals, dir_path_permutation, CFG=CFG)

    #==============================================================
    dir_path = class_path + 'permutation_importance/'
    percentages = [70,80]
    for percentage in percentages:
        variables_element = {}
        for element in ['Nitrogen', 'Phosphorus', 'Potassium']:
            # Ruta de variables más importantes en cada algoritmo
            #csv_path = f"{CFG.class_path}best_{percentage}_percent_features_weight_3_{element}.csv"
            csv_path = f"{dir_path}best_{percentage}_percent_features_{element}.csv"
            # Obtener las más importantes ponderadas. Para el modelo específico
            vars = most_frequent_variables_analysis(csv_path, 
                                                    element=element, 
                                                    percentage=percentage,
                                                    dir_path=dir_path)
            variables_element[element] = vars

        # Guardar como JSON
        with open(f"{dir_path}" + f'most_frequent_variables_{percentage}.json', 'w') as f:
            json.dump(variables_element, f, indent=4)

        # guardar en dataframe csv
        df_vars = pd.DataFrame.from_dict(variables_element, orient='index').transpose()
        df_vars.to_csv(dir_path + f'most_frequent_variables_TOTAL_{percentage}.csv', index=False)

def permutation_importance_Quartiles(all_results, df_imputed, dir_path_permutation, class_path, CFG, all_models = False, best_variables = None):
    
    os.makedirs(dir_path_permutation, exist_ok=True)
    dict_perm_impt_vals = permutation_importance_all_elements(all_results, df_imputed, dir_path_permutation,
                                                              list_elements = ['Quartiles'], CFG=CFG, all_models=all_models, best_variables=best_variables)
    most_frequent_variables_analysis_PERMUTATION(df_imputed, dict_perm_impt_vals, dir_path_permutation, CFG=CFG, best_variables=best_variables)

    #==============================================================
    dir_path = class_path + 'permutation_importance/'
    
    percentages = [70,80]
    for percentage in percentages:
        variables_element = {}
    
        # Ruta de variables más importantes en cada algoritmo
        #csv_path = f"{CFG.class_path}best_{percentage}_percent_features_weight_2_Quartiles.csv"
        csv_path = f"{dir_path}best_{percentage}_percent_features_Quartiles.csv"
        # Obtener las más importantes ponderadas. Para el modelo específico
        vars = most_frequent_variables_analysis(csv_path, 
                                                element='Quartiles', 
                                                percentage=percentage,
                                                dir_path=dir_path)
        variables_element['Quartiles'] = vars

        # Guardar como JSON
        with open(f'{dir_path}most_frequent_variables_{percentage}.json', 'w') as f:
            json.dump(variables_element, f, indent=4)

        # guardar en dataframe csv
        df_vars = pd.DataFrame.from_dict(variables_element, orient='index').transpose()
        df_vars.to_csv(f'{dir_path}most_frequent_variables_TOTAL_{percentage}.csv', index=False)
