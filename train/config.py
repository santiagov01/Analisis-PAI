# Sklearn imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier



class CFG:
    colab = False  # Cambiar a True si se usa Colab
    Root = R'D:\Estudio\OneDrive - Universidad de Antioquia\Estudio\PAI\Codigo' if not colab else '/content/drive/MyDrive/PAI'
    path_df_imputed = f'{Root}/BaseDatos/df_imputed_with_original.csv'
    path_df_imputed_corrected = f'{Root}/BaseDatos/df_imputed_corrected.csv'
    elements_list = ["Nitrogen", "Phosphorus", "Potassium"]
    productivity_vars = ["Plant_Height (cm)", "Number of Flowers", 'Number of Harvested Fruits', 
                         'Weight of Harvested Fruits (Kg)','Fruit Height (mm)', 'Fruit Diameter (mm)']
    model_list = ['RF', 'SVM', 'MLP', 'KNN']
    include_prod = False  # Para incluir variables de productividad


    # NOTE: Cambiar el siguiente flag segun el tipo de entrenamiento
    individual_train = True # Para entrenar con los elementos por separado
    cuartiles_train = False

    # NOTE: Cambiar el siguiente path segun include prod
    if include_prod:
        if cuartiles_train:
            class_path = f'{Root}/Resultados/classification_cuartiles_include_prod/'
        else:
            class_path = f'{Root}/Resultados/classification_include_prod/'
    else:
        if cuartiles_train:
            class_path = f'{Root}/Resultados/classification_cuartiles_exclude_prod/'
        else:
            class_path = f'{Root}/Resultados/classification_exclude_prod/'


    if individual_train:
        path_pkl_results_classification = f"{class_path}class_results_individual_elements.pkl"
    elif cuartiles_train:
        path_pkl_results_classification = f"{class_path}class_models_cuartiles.pkl"
    else:
        path_pkl_results_classification = f"{class_path}all_classification_models.pkl"

    treat_quantiles_path = f'{Root}/Resultados/treatments_quantile_unified.json'

# Configuración de modelos
MODELS_CONFIG = {
    'RF': {
        'estimator': RandomForestClassifier(random_state=42),
        'param_grid': {
            'clf__n_estimators': [50, 100, 200, 300],
            'clf__max_depth': [10, 20 , 30, 40, 50],
        },
        'model_type': 'tree'
    },
    'SVM': {
        'estimator': SVC(probability=True, random_state=42),
        'param_grid': {
            'clf__C': [0.1, 1, 10, 100, 300],
            'clf__kernel': ['rbf'],
            'clf__gamma': ['auto', 0.01, 0.1, 1, 10]
        },
        'model_type': 'kernel'
    },
    'KNN': {
        'estimator': KNeighborsClassifier(),
        'param_grid': {
            'clf__n_neighbors': [2, 3, 5, 7, 9]
        },
        'model_type': 'kernel'
    },
    'MLP': {
        'estimator': MLPClassifier(max_iter=500, random_state=42),
        'param_grid': {
            'clf__hidden_layer_sizes': [(50,), (100,), (200,), (100, 50)],
            'clf__alpha': [0.00001, 0.0001, 0.001]
        },
        'model_type': 'kernel'
    }
    ,
    'XGB': {
        'estimator': XGBClassifier(
            random_state=42,
            eval_metric='mlogloss' # Ojo cambiar a binary:logistic si es binario
        ),
        'param_grid': {
            'clf__n_estimators': [200, 300, 400],
            'clf__max_depth': [5, 7, 10],
            'clf__learning_rate': [0.01, 0.1, 0.2, 0.3],
            'clf__subsample': [0.8, 1.0],
            'clf__colsample_bytree': [0.8, 1.0]
        },
        'model_type': 'tree'
    }
        
}

# Rango de número de clases a evaluar
N_CLASES_RANGE = range(2, 10)