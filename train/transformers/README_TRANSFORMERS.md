# Transformer Models Implementation

Este documento describe la implementación de modelos basados en transformers para clasificación de datos tabulares, incluyendo SwiFT, TTL y TabNet.

## 📋 Resumen de Archivos Creados/Modificados

### Archivos Nuevos

1. **`config_transformers.py`**
   - Configuración de modelos transformer (SwiFT, TTL, TabNet)
   - Grids de hiperparámetros para GridSearchCV
   - Configuración rápida para pruebas

2. **`merge_results_utils.py`**
   - Funciones para unir resultados de diferentes tipos de modelos
   - Verificación de estructura de resultados
   - Creación de comparativas y resúmenes

3. **`merge_all_results.py`**
   - Script de ejemplo para unir resultados
   - Comparación entre modelos tradicionales y transformers

### Archivos Modificados

1. **`utils.py`**
   - Agregadas importaciones de PyTorch y TabNet
   - Implementadas clases wrapper:
     - `TabNetClassifierWrapper`
     - `SwiFTClassifier` y `SwiFTModel`
     - `TTLClassifier` y `TTLModel`
   - Nueva función: `train_test_transformers()`
   - Nueva función: `build_transformer_pipeline()`

2. **`train_NPK_transformers.py`**
   - Completamente actualizado para usar modelos transformer
   - Estructura similar a scripts de entrenamiento tradicionales
   - Flujo completo: entrenamiento → SHAP → Permutation Importance

## 🏗️ Arquitecturas Implementadas

### 1. TabNet
- **Descripción**: Arquitectura de atención tabular con aprendizaje sparse
- **Ventajas**: Interpretabilidad nativa, feature importance automática
- **Hiperparámetros principales**:
  - `n_d`, `n_a`: Dimensiones de atención
  - `n_steps`: Pasos de decisión
  - `lambda_sparse`: Regularización sparse

### 2. SwiFT (Sparse Weighted Fusion Transformer)
- **Descripción**: Transformer con fusión sparse para datos tabulares
- **Ventajas**: Captura relaciones complejas, eficiente en memoria
- **Hiperparámetros principales**:
  - `d_model`: Dimensión del modelo
  - `nhead`: Número de cabezas de atención
  - `num_layers`: Capas del transformer

### 3. TTL (Transformer-based Tabular Learning)
- **Descripción**: Transformer optimizado con Pre-LN architecture
- **Ventajas**: Mejor convergencia, arquitectura moderna
- **Hiperparámetros principales**:
  - `d_model`: Dimensión del modelo (mayor que SwiFT)
  - `nhead`: Número de cabezas de atención
  - `num_layers`: Capas del transformer

## 📂 Estructura de Archivos

```
train/
│
├── config.py                          # Configuración tradicional
├── config_transformers.py             # ⭐ Configuración transformers
├── utils.py                           # ⭐ Funciones principales (modificado)
│
├── train_NPK_transformers.py          # ⭐ Script principal transformers
├── merge_all_results.py               # ⭐ Script para unir resultados
├── merge_results_utils.py             # ⭐ Utilidades de merge
│
└── train_NPK.py                       # Script tradicional (referencia)
```

## 🚀 Uso del Sistema

### Paso 1: Instalar Dependencias

```bash
pip install torch
pip install pytorch-tabnet
pip install scikit-learn
pip install pandas numpy matplotlib seaborn
pip install shap
```

### Paso 2: Entrenar Modelos Transformer

```python
# Ejecutar el script de entrenamiento
python train_NPK_transformers.py
```

Este script realizará:
1. ✅ Entrenamiento de SwiFT, TTL y TabNet
2. ✅ Validación cruzada con GridSearchCV
3. ✅ Cálculo de métricas (Accuracy, F1, Precision, Recall)
4. ✅ Generación de matrices de confusión
5. ✅ Cálculo de valores SHAP
6. ✅ Cálculo de Permutation Importance
7. ✅ Análisis de variables más importantes

### Paso 3: Unir Resultados con Modelos Tradicionales

```python
# Ejecutar el script de merge
python merge_all_results.py
```

O usar directamente las funciones:

```python
from merge_results_utils import merge_and_save_all

merged = merge_and_save_all(
    traditional_path='../Resultados/classification_exclude_prod/class_results_individual_elements.pkl',
    transformer_path='../Resultados/classification_exclude_prod_transformer/class_results_individual_elements_transformer.pkl',
    output_path='../Resultados/classification_exclude_prod/class_results_all_models_combined.pkl'
)
```

## 📊 Estructura de Resultados

Los resultados se guardan con la misma estructura que los modelos tradicionales:

```python
{
    'SwiFT': [
        {
            'n_clases': '3_Nitrogen',
            'model_name': 'SwiFT',
            'accuracy_train': 0.85,
            'accuracy_test': 0.82,
            'f1_train': 0.84,
            'f1_test': 0.81,
            'best_params': {...},
            'class_distribution': {...},
            'classification_report': {...},
            'confusion_matrix_test': [...],
            'shap_values': [...],
            'X_scaled_df': DataFrame(...)
        },
        # ... más elementos (Phosphorus, Potassium)
    ],
    'TTL': [...],
    'TabNet': [...]
}
```

## 🔍 Diferencias Clave con Modelos Tradicionales

### 1. Pipeline
- **Tradicionales**: `StandardScaler` → `SMOTE` (opcional) → `Classifier`
- **Transformers**: `SMOTE` (opcional) → `Classifier` (escalado interno)

### 2. SHAP Values
- **Tradicionales**: TreeExplainer o KernelExplainer según tipo
- **Transformers**: Siempre KernelExplainer (más lento pero funciona para todos)

### 3. Entrenamiento
- **Tradicionales**: Paralelo (`n_jobs=-1`)
- **Transformers**: Secuencial (ya optimizan GPU/CPU internamente)

### 4. Hiperparámetros
- **Tradicionales**: Grids pequeños, rápidos
- **Transformers**: Grids más grandes, más tiempo de entrenamiento

## 📁 Jerarquía de Directorios

Los transformers mantienen la misma estructura de carpetas con sufijo `_transformer`:

```
Resultados/
│
├── classification_exclude_prod/          # Modelos tradicionales
│   ├── RF/
│   ├── SVM/
│   ├── models/
│   └── class_results_individual_elements.pkl
│
├── classification_exclude_prod_transformer/   # ⭐ Modelos transformer
│   ├── SwiFT/
│   ├── TTL/
│   ├── TabNet/
│   ├── models/
│   └── class_results_individual_elements_transformer.pkl
│
└── classification_exclude_prod/          # Resultados combinados
    └── class_results_all_models_combined.pkl
```

## 🎯 Funciones Principales

### En `utils.py`

#### `train_test_transformers()`
Función principal para entrenar modelos transformer.

```python
resultados = train_test_transformers(
    df_imputed=df_imputed,
    n_clases=None,
    model_name='SwiFT',
    model_config=TRANSFORMERS_CONFIG['SwiFT'],
    element='Nitrogen',
    usar_smote=False,
    mostrar_graficos=True,
    calcular_shap=True,
    dir_path='../Resultados/classification_exclude_prod_transformer/SwiFT/',
    CFG=CFG
)
```

**Características**:
- ✅ Compatible con la estructura existente
- ✅ Manejo automático de SHAP con KernelExplainer
- ✅ Grid Search con validación cruzada
- ✅ Guardado de modelos y resultados

### En `merge_results_utils.py`

#### `merge_model_results()`
Une dos diccionarios de resultados.

```python
merged = merge_model_results(
    results_dict1=traditional_results,
    results_dict2=transformer_results,
    verify_structure=True
)
```

#### `print_comparison_summary()`
Imprime un resumen comparativo de todos los modelos.

```python
print_comparison_summary(merged_results)
```

## ⚠️ Notas Importantes

1. **Tiempo de Entrenamiento**: Los transformers son más lentos que modelos tradicionales, especialmente con grids grandes.

2. **Memoria**: TabNet y transformers requieren más memoria. Considerar usar `TRANSFORMERS_CONFIG_QUICK` para pruebas rápidas.

3. **GPU**: Los transformers pueden usar GPU automáticamente si está disponible (PyTorch detecta CUDA).

4. **SHAP**: El cálculo de SHAP para transformers es más lento. Se usa muestra reducida (100 puntos) por defecto.

5. **Nombres de Modelos**: Mantener consistencia:
   - Tradicionales: 'RF', 'SVM', 'KNN', 'MLP', 'XGB'
   - Transformers: 'SwiFT', 'TTL', 'TabNet'

## 🔄 Flujo Completo Recomendado

1. **Entrenar modelos tradicionales** (opcional, si no se ha hecho):
   ```bash
   python train_NPK.py
   ```

2. **Entrenar modelos transformer**:
   ```bash
   python train_NPK_transformers.py
   ```

3. **Unir resultados**:
   ```bash
   python merge_all_results.py
   ```

4. **Analizar** resultados combinados usando las funciones existentes:
   - `compare_classification_models()`
   - `permutation_importance_NPK()`
   - `most_frequent_variables_analysis()`

## 📝 Ejemplo de Uso Completo

```python
from utils import *
from config_transformers import TRANSFORMERS_CONFIG
from merge_results_utils import merge_and_save_all

# 1. Cargar datos
df_imputed = pd.read_csv(CFG.path_df_imputed_corrected)
df_imputed.columns = clean_feature_names(df_imputed.columns)

# 2. Configurar
CFG.class_path = f'{CFG.Root}/Resultados/classification_exclude_prod_transformer/'
CFG.individual_train = True

# 3. Entrenar un modelo transformer
resultado = train_test_transformers(
    df_imputed=df_imputed,
    n_clases=None,
    model_name='TabNet',
    model_config=TRANSFORMERS_CONFIG['TabNet'],
    element='Nitrogen',
    usar_smote=False,
    calcular_shap=True,
    dir_path=f"{CFG.class_path}TabNet/",
    CFG=CFG
)

# 4. Unir con resultados tradicionales
merged = merge_and_save_all(
    traditional_path='path/to/traditional.pkl',
    transformer_path='path/to/transformer.pkl',
    output_path='path/to/combined.pkl'
)
```

## 🎓 Referencias

- **TabNet**: https://arxiv.org/abs/1908.07442
- **Transformers**: https://arxiv.org/abs/1706.03762
- **PyTorch**: https://pytorch.org/
- **SHAP**: https://github.com/slundberg/shap

## ✅ Checklist de Implementación

- [x] Implementar clases wrapper para transformers
- [x] Crear función `train_test_transformers()`
- [x] Configurar hiperparámetros en `config_transformers.py`
- [x] Adaptar flujo SHAP para transformers
- [x] Adaptar flujo Permutation Importance
- [x] Crear funciones de merge
- [x] Mantener consistencia de nombres y estructura
- [x] Documentar sistema completo

## 🤝 Contribución

Para añadir nuevos modelos transformer:

1. Crear clase wrapper en `utils.py` con métodos:
   - `fit(X, y)`
   - `predict(X)`
   - `predict_proba(X)`
   - `get_params()` y `set_params()`

2. Agregar configuración en `config_transformers.py`

3. El sistema lo integrará automáticamente.


