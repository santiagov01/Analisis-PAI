
# Sklearn imports
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split, cross_validate
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score, precision_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.base import clone

import pandas as pd
import numpy as np

import json
import pickle

import os

from matplotlib import pyplot as plt
import seaborn as sns

from config import CFG


def split_data_NPK(df, test_size=0.3, random_state=None):
    """Partición estratificada por tratamiento NPK con semilla configurable."""
    df_temp = df.copy()
    df_temp['Etiqueta_NPK'] = df_temp['Tratamiento'].str.extract(r'(N\dP\dK\d)')
    
    train_data, test_data = train_test_split(
        df_temp,
        test_size=test_size,
        random_state=random_state,
        stratify=df_temp['Etiqueta_NPK']
    )
    return train_data.drop(columns=['Etiqueta_NPK']), test_data.drop(columns=['Etiqueta_NPK'])

def renombrar_columnas(df):
    mapeo = {
    "Altura planta (cm)": "Plant_Height (cm)",
    "Clorofila (SPAD)": "Chlorophyll (SPAD)",
    "Numero flores ": "Number of Flowers",
    "Numero frutos cosechados ": "Number of Harvested Fruits",
    "Peso frutos cosechados (Kg)": "Weight of Harvested Fruits (Kg)",
    "Tamanno altura (mm)": "Fruit Height (mm)",
    "pH_savia": "Sap pH",
    "K_savia (ppm)": "Sap K (ppm)",
    "Ca_savia (ppm)": "Sap Ca (ppm)",
    "Na_savia (ppm)": "Sap Na (ppm)",
    "NO3_savia (ppm)": "Sap NO3 (ppm)",
    "Conductividad_savia (mS/cm)": "Sap Conductivity (mS/cm)",
    "pH_suelo_Horiba": "Soil pH Horiba",
    "K_suelo_Horiba (ppm)": "Soil K Horiba (ppm)",
    "Ca_suelo_Horiba (ppm)": "Soil Ca Horiba (ppm)",
    "Na_suelo_Horiba (ppm)": "Soil Na Horiba (ppm)",
    "NO3_suelo_Horiba (ppm)": "Soil NO3 Horiba (ppm)",
    "Conductividad_suelo_Horiba (mS/cm)": "Soil Conductivity Horiba (mS/cm)",
    "Tamanno cintura (mm)": "Fruit Diameter (mm)",
    "N": "Nitrogen",
    "P": "Phosphorus",
    "K": "Potassium",
    "Año": "Year",
    "Mes": "Month",
    "Día": "Day",
    "Tratamiento_num": "Treatment_Num"
    }
    return df.rename(columns=mapeo)

def impute_data(train_data, test_data, seed=42, columnas_productividad=None):

    # Por defecto, si no se pasa nada, es una lista vacía
    #if columnas_productividad is None:
    columnas_productividad = ['Numero frutos cosechados ',
                                  'Peso frutos cosechados (Kg)', 
                                 'Tamanno altura (mm)', 
                                 'Tamanno cintura (mm)', 
                                 'Numero flores ']

        
    train = train_data.copy()
    test = test_data.copy()

    # --- NOTE: NUEVA REGLA: Llenar con 0 los nulos de productividad en train y test ---
    for col in columnas_productividad:
        if col in train.columns:
            train[col] = train[col].fillna(0)
        if col in test.columns:
            test[col] = test[col].fillna(0)

    # 1. Extraer tratamiento
    train['Tratamiento_num'] = train['Tratamiento'].str.extract(r'T(\d+)').astype(float)
    test['Tratamiento_num'] = test['Tratamiento'].str.extract(r'T(\d+)').astype(float)

    # 2. Seleccionar columnas SOLO basados en el Train
    min_frac_no_nulos = 0.05
    df_train_num = train.select_dtypes(include=[float, int])
    
    # Calcular columnas válidas en train
    columnas_validas = df_train_num.columns[df_train_num.isna().mean() < (1 - min_frac_no_nulos)]
    df_train_num = df_train_num[columnas_validas]
    
    # Forzar al test a tener EXACTAMENTE las mismas columnas (rellenando con NaN si falta alguna)
    df_test_num = pd.DataFrame(index=test.index)
    for col in columnas_validas:
        df_test_num[col] = test[col] if col in test.columns else np.nan

    # 3. Límites de clipping (Solo Train)
    limites_clip = df_train_num.quantile([0.01, 0.99]).T
    limites_clip.columns = ['min_val', 'max_val']

    df_train_original = df_train_num.copy()
    df_test_original = df_test_num.copy()

    df_train_imputed = df_train_num.copy()
    df_test_imputed = df_test_num.copy()

    # 4. Iterar por tratamientos conocidos en el Train
    for tratamiento in df_train_num['Tratamiento_num'].dropna().unique():
        mask_train = df_train_num['Tratamiento_num'] == tratamiento
        mask_test = df_test_num['Tratamiento_num'] == tratamiento

        grupo_train = df_train_num.loc[mask_train].drop(columns=['Tratamiento_num'])
        grupo_test = df_test_num.loc[mask_test].drop(columns=['Tratamiento_num'])

        # Solo imputar si hay columnas numéricas válidas para este grupo
        cols_imputar = grupo_train.columns[~grupo_train.isna().all()]
        if len(cols_imputar) == 0:
            continue
            
        # Instanciar modelo con min_value=0 para evitar negativos automáticamentee
        imputer = IterativeImputer(
            estimator=RandomForestRegressor(n_estimators=10, random_state=seed),
            max_iter=20, random_state=seed, min_value=0
        )

        # FIT_TRANSFORM en Train
        imputado_train = imputer.fit_transform(grupo_train[cols_imputar])
        df_train_imputed.loc[mask_train, cols_imputar] = imputado_train
        
        # Restaurar originales no nulos en Train
        mascara_orig_train = ~grupo_train[cols_imputar].isna()
        df_train_imputed.loc[mask_train, cols_imputar] = df_train_imputed.loc[mask_train, cols_imputar].where(~mascara_orig_train, grupo_train[cols_imputar])

        # TRANSFORM en Test (si hay datos para este tratamiento)
        if mask_test.any():
            imputado_test = imputer.transform(grupo_test[cols_imputar])
            df_test_imputed.loc[mask_test, cols_imputar] = imputado_test
            
            # Restaurar originales no nulos en Test
            mascara_orig_test = ~grupo_test[cols_imputar].isna()
            df_test_imputed.loc[mask_test, cols_imputar] = df_test_imputed.loc[mask_test, cols_imputar].where(~mascara_orig_test, grupo_test[cols_imputar])

    # 5. Aplicar Clipping SOLO a los valores que eran NaN originalmente
    for col in df_train_imputed.columns:
        if col in limites_clip.index and col != 'Tratamiento_num':
            min_v, max_v = limites_clip.loc[col, 'min_val'], limites_clip.loc[col, 'max_val']
            
            # Clip en Train
            mask_imp_train = df_train_original[col].isna()
            df_train_imputed.loc[mask_imp_train, col] = df_train_imputed.loc[mask_imp_train, col].clip(lower=min_v, upper=max_v)
            
            # Clip en Test
            mask_imp_test = df_test_original[col].isna()
            df_test_imputed.loc[mask_imp_test, col] = df_test_imputed.loc[mask_imp_test, col].clip(lower=min_v, upper=max_v)

    # 6. Renombrar columnas a inglés
    df_train_imputed = renombrar_columnas(df_train_imputed)
    df_test_imputed = renombrar_columnas(df_test_imputed)
    return df_train_imputed, df_test_imputed

def build_pipeline(model_config, seed):
    pipeline_steps = [
        ('scaler', StandardScaler()),
        ('clf', clone(model_config['estimator']))
    ]
    return Pipeline(pipeline_steps)

def preparar_datos_NPK(df_train_imputed, df_test_imputed, element):
    if element not in CFG.elements_list:
        raise ValueError(f"Elemento '{element}' no válido. Debe ser uno de {CFG.elements_list}.")
    df_train_imputed['target'] = df_train_imputed[element]
    df_test_imputed['target'] = df_test_imputed[element]

        # Eliminar columnas no necesarias
    columns_to_drop = ['Nitrogen', 'Phosphorus', 'Potassium', 'target',
                       'Clase_custom', 'Treatment_Num', 'Year', 'Month', 'Day']

    X_train = df_train_imputed.drop(columns=columns_to_drop, errors='ignore')
    y_train = df_train_imputed['target']

    X_test = df_test_imputed.drop(columns=columns_to_drop, errors='ignore')
    y_test = df_test_imputed['target']

    feature_names = X_train.columns.tolist()

    return X_train, X_test, y_train, y_test, feature_names

def calculate_cv_metrics(grid, X_train, y_train, cv):
        # Métricas train del mejor modelo (usando cross-validation)
    nested_score = cross_validate(
        grid.best_estimator_, X=X_train, y=y_train,
        cv=cv, scoring=['f1_micro', 'f1_macro', 'accuracy'],
        return_train_score=True
    )
    acc_train = np.mean(nested_score['train_accuracy'])
    f1_train = np.mean(nested_score['train_f1_micro'])
    f1_train_macro = np.mean(nested_score['train_f1_macro'])

    return acc_train, f1_train, f1_train_macro

def return_classification_metrics(y_test, y_test_pred):
    acc_test = accuracy_score(y_test, y_test_pred)
    f1_test = f1_score(y_test, y_test_pred, average='micro')
    f1_test_macro = f1_score(y_test, y_test_pred, average='macro')
    return acc_test, f1_test, f1_test_macro

def train_test_NPK(
        df_train_imputed,
        df_test_imputed,
        element,
        model_name,
        model_config,
        seed=42
    ):
    # Preparar datos
    X_train, X_test, y_train, y_test, feature_names = preparar_datos_NPK(
        df_train_imputed, df_test_imputed, element
    )

    pipe = build_pipeline(model_config, seed=42)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    grid_search = GridSearchCV(
        estimator=pipe,
        param_grid=model_config['param_grid'],
        cv=cv,
        scoring='f1_macro',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    acc_train, f1_train, f1_train_macro = calculate_cv_metrics(grid_search, 
                                                               X_train, y_train,
                                                               cv)
    #=========== TEST ================
    y_pred_test = grid_search.predict(X_test)
    acc_test, f1_test, f1_test_macro = return_classification_metrics(y_test, y_pred_test)

    class_report_dict = classification_report(y_test, y_pred_test, output_dict=True)
    cm_test = confusion_matrix(y_test, y_pred_test)

    n_clases_str = f'{3}_{element}'

    resultados = {
        'y_true': y_test.tolist(),
        'y_pred': y_pred_test.tolist(),
        'y_test': y_test,
        'X_train': X_train,
        'X_test': X_test,
        'model_name': model_name,
        'acc_train': acc_train,
        'f1_train': f1_train,
        'f1_train_macro': f1_train_macro,
        'acc_test': acc_test,
        'f1_test': f1_test,
        'f1_test_macro': f1_test_macro,
        'best_params': grid_search.best_params_,
        'classification_report': class_report_dict,
        'confusion_matrix_test': cm_test,
        'grid_search': grid_search,
        'feature_names': feature_names,
        'n_clases': n_clases_str
    }
    return resultados

def save_results_iterations(historico_resultados, best_results, mejores_iteraciones_info,
                            base_path="./Resultados/pipeline_imputation_NPK"):
    # Guardar resultados completos de todas las iteraciones
    with open(f'{base_path}/historico_resultados.pkl', 'wb') as f:
        pickle.dump(historico_resultados, f)

    # Guardar resultados de la mejor iteración para cada elemento
    with open(f'{base_path}/best_results.pkl', 'wb') as f:
        pickle.dump(best_results, f)

    # Guardar metadatos de la mejor iteración para cada elemento en JSON
    with open(f'{base_path}/best_iterations_info.json', 'w') as f:
        json.dump(mejores_iteraciones_info, f, indent=4)

def extract_frequent_values_from_csv(csv_path: str, threshold_percentage: float = 80) -> list:
    df = pd.read_csv(csv_path)

    column_rankings = []
    column_sets = []
    for col in df.columns:
        rank_dict = {}
        for pos, val in enumerate(df[col]):
            if pd.notna(val) and str(val).strip() != "":
                rank_dict[val] = pos
        column_rankings.append(rank_dict)
        column_sets.append(set(rank_dict.keys()))

    if not column_sets:
        return []

    num_columns = len(column_sets)
    min_appearances = int(np.ceil(num_columns * threshold_percentage / 100))

    all_values = set()
    for col_set in column_sets:
        all_values.update(col_set)

    frequent_values = set()
    for val in all_values:
        count = sum(1 for col_set in column_sets if val in col_set)
        if count >= min_appearances:
            frequent_values.add(val)

    if not frequent_values:
        return []

    scores = {}
    for val in frequent_values:
        scores[val] = sum(rank_dict[val] for rank_dict in column_rankings if val in rank_dict)

    sorted_values = sorted(scores.keys(), key=lambda v: scores[v])

    return sorted_values