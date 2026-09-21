import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import eli5
from eli5.sklearn import PermutationImportance

def codificar_cuartiles(df, treat_quantiles_path):
    """Filtra y mapea tratamientos a clases binarias (0: Q1 baja prod, 1: Q4 alta prod)."""
    with open(treat_quantiles_path, 'r') as f:
        quantiles_dict = json.load(f)
    
    treatment_to_class = {
        int(t): int(clase) 
        for clase, treatments in quantiles_dict.items() 
        for t in treatments
    }
    
    df_out = df.copy()
    if 'Treatment_Num' not in df_out.columns and 'Tratamiento' in df_out.columns:
        df_out['Treatment_Num'] = df_out['Tratamiento'].str.extract(r'T(\d+)').astype(float)
        
    df_filtered = df_out[df_out['Treatment_Num'].isin(treatment_to_class.keys())].copy()
    y = df_filtered['Treatment_Num'].map(treatment_to_class).astype(int)
    return df_filtered, y

def preparar_datos_cuartiles(df_train, df_test, treat_quantiles_path, productivity_vars=None, include_prod=False):
    """Prepara X e y para train y test eliminando metadatos y fugas de productividad."""
    train_f, y_train = codificar_cuartiles(df_train, treat_quantiles_path)
    test_f, y_test = codificar_cuartiles(df_test, treat_quantiles_path)
    
    cols_drop = ['Nitrogen', 'Phosphorus', 'Potassium', 'target', 'Clase_custom', 
                 'Treatment_Num', 'Tratamiento', 'Year', 'Month', 'Day', 'Etiqueta_NPK']
    if not include_prod and productivity_vars:
        cols_drop += productivity_vars
        
    X_train = train_f.drop(columns=cols_drop, errors='ignore').select_dtypes(include=[np.number])
    X_test = test_f[X_train.columns].copy()
    
    return X_train, X_test, y_train, y_test, list(X_train.columns)

def train_test_cuartiles(train_data, test_data, model_name, model_config, 
                         treat_quantiles_path, productivity_vars=None, 
                         include_prod=False, seed=42):
    """Entrena y evalúa un modelo de clasificación para cuartiles."""
    X_train, X_test, y_train, y_test, feature_names = preparar_datos_cuartiles(
        train_data, test_data, treat_quantiles_path, productivity_vars, include_prod
    )
    
    estimator = clone(model_config['estimator'])
    if hasattr(estimator, 'random_state'):
        estimator.set_params(random_state=seed)
    if model_name == 'XGB':
        estimator.set_params(objective='binary:logistic', eval_metric='logloss')

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', estimator)
    ])
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    grid = GridSearchCV(pipe, model_config['param_grid'], cv=cv, scoring='f1_macro', n_jobs=-1)
    grid.fit(X_train, y_train)
    
    y_pred = grid.predict(X_test)
    
    return {
        'model_name': model_name,
        'accuracy_test': accuracy_score(y_test, y_pred),
        'f1_test_macro': f1_score(y_test, y_pred, average='macro'),
        'best_params': grid.best_params_,
        'grid_search': grid,
        'X_test': X_test,
        'y_test': y_test,
        'feature_names': feature_names,
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }

def run_permutation_cuartiles(best_results, models_config, output_dir, seed=42):
    """Ejecuta Permutation Importance para clasificación binaria de cuartiles."""
    os.makedirs(output_dir, exist_ok=True)
    first_model = list(models_config.keys())[0]
    features = best_results[first_model]['feature_names']
    df_perm = pd.DataFrame({'Feature': features})
    best_feats_by_alg = {}

    for model_name in models_config.keys():
        res = best_results[model_name]
        pipeline = res['grid_search'].best_estimator_
        
        perm = PermutationImportance(pipeline, random_state=seed, n_iter=10, cv="prefit").fit(res['X_test'], res['y_test'])
        imp = np.maximum(perm.feature_importances_, 0.0)
        norm_imp = imp / np.sum(imp) if np.sum(imp) > 0 else imp
        df_perm[model_name] = norm_imp
        
        # Selección 80% acumulado
        df_sorted = pd.DataFrame({'Variable': features, 'Imp': norm_imp}).sort_values('Imp', ascending=False)
        df_sorted['Acc'] = df_sorted['Imp'].cumsum()
        top_vars = df_sorted[df_sorted['Acc'] <= 0.80]['Variable'].tolist()
        best_feats_by_alg[model_name] = top_vars if top_vars else df_sorted['Variable'].iloc[:1].tolist()

    df_perm.to_csv(os.path.join(output_dir, "permutation_importance_cuartiles.csv"), index=False)
    df_top = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in best_feats_by_alg.items()]))
    df_top.to_csv(os.path.join(output_dir, "best_80_percent_features_cuartiles.csv"), index=False)
    
    # Consenso por algoritmo: calcular los niveles 80% y 100%
    counts = pd.Series(
        feature
        for variables in best_feats_by_alg.values()
        for feature in variables
    ).value_counts()
    consensus = {}
    for percentage in (80, 100):
        min_models = int(np.ceil(percentage / 100 * len(best_feats_by_alg)))
        consensus[percentage] = counts[counts >= min_models].index.tolist()
        pd.DataFrame({
            'Variables': consensus[percentage]
        }).to_csv(
            os.path.join(output_dir, f'common_{percentage}_percent_features_cuartiles.csv'),
            index=False
        )

    return consensus