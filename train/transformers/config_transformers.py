# Configuration for Transformer-based models
# This file contains model configurations for SwiFT, TTL, and TabNet

from utils import TabNetClassifierWrapper, SwiFTClassifier, TTLClassifier

# Configuración de modelos transformer
TRANSFORMERS_CONFIG = {
    'TabNet': {
        'estimator': TabNetClassifierWrapper(
            seed=42,
            verbose=0,
            device_name='auto'
        ),
        'param_grid': {
            'clf__n_d': [8, 16, 32],
            'clf__n_a': [8, 16, 32],
            'clf__n_steps': [3, 4, 5],
            'clf__gamma': [1.0, 1.3, 1.5],
            'clf__lambda_sparse': [1e-4, 1e-3, 1e-2]
        },
        'model_type': 'kernel'  # Para SHAP usamos KernelExplainer
    },
    'SwiFT': {
        'estimator': SwiFTClassifier(
            seed=42
        ),
        'param_grid': {
            'clf__d_model': [32, 64, 128],
            'clf__nhead': [2, 4, 8],
            'clf__num_layers': [2, 3, 4],
            'clf__dim_feedforward': [64, 128, 256],
            'clf__dropout': [0.1, 0.2, 0.3],
            'clf__max_epochs': [50, 100],
            'clf__batch_size': [32, 64],
            'clf__lr': [1e-4, 1e-3, 1e-2]
        },
        'model_type': 'kernel'  # Para SHAP usamos KernelExplainer
    },
    'TTL': {
        'estimator': TTLClassifier(
            seed=42
        ),
        'param_grid': {
            'clf__d_model': [64, 128, 256],
            'clf__nhead': [4, 8, 16],
            'clf__num_layers': [2, 3, 4],
            'clf__dim_feedforward': [128, 256, 512],
            'clf__dropout': [0.1, 0.15, 0.2],
            'clf__max_epochs': [50, 100, 150],
            'clf__batch_size': [16, 32, 64],
            'clf__lr': [1e-4, 1e-3, 5e-3]
        },
        'model_type': 'kernel'  # Para SHAP usamos KernelExplainer
    }
}

# Configuración reducida para pruebas rápidas (opcional)
TRANSFORMERS_CONFIG_QUICK = {
    'TabNet': {
        'estimator': TabNetClassifierWrapper(
            seed=42,
            verbose=0,
            device_name='auto'
        ),
        'param_grid': {
            'clf__n_d': [16],
            'clf__n_a': [16],
            'clf__n_steps': [3],
            'clf__gamma': [1.3],
            'clf__lambda_sparse': [1e-3]
        },
        'model_type': 'kernel'
    },
    'SwiFT': {
        'estimator': SwiFTClassifier(
            seed=42
        ),
        'param_grid': {
            'clf__d_model': [64],
            'clf__nhead': [4],
            'clf__num_layers': [2],
            'clf__dim_feedforward': [128],
            'clf__dropout': [0.1],
            'clf__max_epochs': [50],
            'clf__batch_size': [64],
            'clf__lr': [1e-3]
        },
        'model_type': 'kernel'
    },
    'TTL': {
        'estimator': TTLClassifier(
            seed=42
        ),
        'param_grid': {
            'clf__d_model': [128],
            'clf__nhead': [8],
            'clf__num_layers': [3],
            'clf__dim_feedforward': [256],
            'clf__dropout': [0.15],
            'clf__max_epochs': [100],
            'clf__batch_size': [32],
            'clf__lr': [1e-3]
        },
        'model_type': 'kernel'
    }
}
