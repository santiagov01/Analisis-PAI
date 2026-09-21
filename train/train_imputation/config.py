# Sklearn imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

import logging
import os

def setup_logger(name="PAI_NPK", log_file=None, level=logging.INFO):
    """Configura un logger profesional con salida a consola y opcionalmente a archivo."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        formatter = logging.Formatter(
            fmt='%(asctime)s - [%(levelname)s] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Salida por consola
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # Salida a archivo de log si se especifica
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
    return logger


class CFG:
    colab = False  # Cambiar a True si se usa Colab
    #Root = '/home/student/PAI/Analisis-PAI' if not colab else '/content/drive/MyDrive/PAI'
    Root = r'D:\Estudio\OneDrive - Universidad de Antioquia\Estudio\PAI\Codigo\Code_Quindio\Analisis-PAI' if not colab else '/content/drive/MyDrive/PAI'
        
    path_df_imputed = f'{Root}/BaseDatos/df_imputed_with_original.csv'
    path_df_imputed_corrected = f'{Root}/BaseDatos/df_imputed_corrected.csv'
    data_path_clean = f'{Root}/BaseDatos/BD Quindio unificados V2/df_filtered.csv'  # Base de datos a particionar
    treat_quantiles_path = f'{Root}/Resultados/treatments_quantile_unified.json'

    elements_list = ["Nitrogen", "Phosphorus", "Potassium"]
    productivity_vars = ["Plant_Height (cm)", "Number of Flowers", 'Number of Harvested Fruits', 
                         'Weight of Harvested Fruits (Kg)','Fruit Height (mm)', 'Fruit Diameter (mm)']
    model_list = ['RF', 'SVM', 'MLP', 'KNN']
    include_prod = False  # Para incluir variables de productividad



    # Directorio de resultados
    results_dir = f'{Root}/Resultados/pipeline_imputation_NPK'
    xai_output_dir = f'{results_dir}/xai_outputs'
    
    # Parámetros del experimento
    elements_list = ["Nitrogen", "Phosphorus", "Potassium"]
    productivity_vars = [
        "Plant_Height (cm)", "Number of Flowers", 'Number of Harvested Fruits', 
        'Weight of Harvested Fruits (Kg)', 'Fruit Height (mm)', 'Fruit Diameter (mm)'
    ]
    
    include_prod = False
    individual_train = True
    cuartiles_train = False
    
    # Parámetros de iteración y reproducibilidad
    n_iterations = 3 #20 iteraciones cambiando porcion de datos de test segun la semilla.
    base_seed = 42
    shap_iterations = 3
    target_metric = 'f1_test_macro'
    test_size = 0.3


# Configuración de modelos
MODELS_CONFIG = {
    'RF': {
        'estimator': RandomForestClassifier(random_state=42),
        'param_grid': {
            'clf__n_estimators': [50],# 100, 200, 300],
            'clf__max_depth': [10], #20 , 30, 40, 50],
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
        'estimator': MLPClassifier(max_iter=500, random_state=42, early_stopping=True),
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
            base_score=0.5,
            eval_metric='mlogloss' # NOTE: Ojo cambiar a binary:logistic si es binario
        ),
        'param_grid': {
            # 'clf__n_estimators': [200],# 300, 400],
            # 'clf__max_depth': [5],# 7, 10],
            # 'clf__learning_rate': [0.3], #[0.01, 0.1, 0.2, 0.3],
            # 'clf__subsample': [0.8], #1.0],
            # 'clf__colsample_bytree': [0.8]#, 1.0]
            'clf__n_estimators': [150, 300],          # 2 opciones bien diferenciadas
            'clf__max_depth': [3, 5, 7],              # Eliminar 10; 3 a 7 es la zona ideal para tabular
            'clf__learning_rate': [0.03, 0.1],        # Un ritmo conservador y uno estándar
            'clf__subsample': [0.8],                  # 0.8 casi siempre supera a 1.0 en generalización
            'clf__colsample_bytree': [0.8],           # 0.8 fuerza diversidad de variables (ideal para XAI)
            'clf__min_child_weight': [1, 3]           # Control fino sobre hojas pequeñas
        },
        'model_type': 'tree'
    }
        
}

# Rango de número de clases a evaluar
N_CLASES_RANGE = range(2, 10)