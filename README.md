# Tomato Fertilization Classification - Quindío

Machine learning pipeline for classifying optimal NPK fertilizer levels in tomato cultivation using multi-source agricultural data from Quindío, Colombia.

## Project Overview

This project develops classification models to predict optimal nitrogen (N), phosphorus (P), and potassium (K) fertilizer levels based on soil properties, plant health indicators, and environmental conditions.

**Key Objectives:**
- Classify fertilizer requirements into deficiency/adequate/excess categories
- Identify most important variables influencing nutrient uptake
- Analyze productivity relationships with fertilizer treatments
- Provide interpretable recommendations using SHAP analysis

## Dataset

**Sources:** Fixed stations (CropX, Fijo), mobile sensors (Movil), manual measurements (Manual), and farm management records (Manejo)

**Variables:**
- Soil: VWC, temperature, EC, pH, nutrients (N, P, K, Ca, Na)
- Plant: height, chlorophyll (SPAD), sap nutrients, flowers, fruit metrics
- Environmental: radiation, temperature, humidity
- Productivity: harvested fruits, weight, fruit dimensions

**Preprocessing:**
- Unified data from 5 sources
- Iterative imputation for missing values
- Removed rows with >24 missing values
- Zero-corrected productivity variables before harvest start date

## Notebooks

### Data Preparation
- **1-Explore_data.ipynb**: Initial data exploration and helper functions
- **2-preprocess_unificar.ipynb**: Merge multiple data sources, plant-treatment mapping
- **3-Preprocess_Analizar_Imputar_Datos.ipynb**: Data cleaning, filtering, and iterative imputation
- **3-2-Eliminar_ceros_BD_Imputed.ipynb**: Correct productivity variables to zero before Oct 2, 2024

### Analysis
- **2-Analyze_Time_Series.ipynb**: Treatment distribution, productivity rankings, cumulative productivity trends

### Model Training
- **4_train.ipynb**: Main classification training (8 classes, quartiles, individual NPK)
- **4_train_less_variables.ipynb**: Train with top 70-80% most important features
- **4_train_PCA.ipynb**: Dimensionality reduction experiments

### Results & Interpretation
- **5-Permutation_Importance.ipynb**: Calculate permutation-based feature importance
- **5-Plot-SHAP-Bar.ipynb**: Generate SHAP bar plots for feature importance
- **5-Results-SHAP_Analysis_Ranking.ipynb**: Comprehensive SHAP analysis and variable ranking
- **6-Results-Productivity-Analysis.ipynb**: Correlation analysis between important variables and productivity

## Training Scripts (`train/`)

**Configuration:**
- `config.py`: Model hyperparameters, paths, global settings
- `utils.py`: Core functions (data prep, model training, SHAP, metrics, visualization)

**Training Modes:**
- `train_NPK.py`: Individual N, P, K classifiers (3 classes each: 0=deficiency, 1=adequate, 2=excess)
- `train_cuartiles.py`: Quartile-based classification (4 productivity levels)
- `train_cuartiles_less_vars.py`: Quartile training with reduced feature set
- `train_cuartiles_all_models.py`: Train using all predictions validation method on quartiles
- `train_NPK_all_models.py`: Train using all predictions validation method in NPK classification
- `train_PCA.py`: PCA-transformed features training

**Key Features:**
- Nested cross-validation for robust evaluation
- GridSearchCV for hyperparameter tuning
- SMOTE support for class imbalance
- SHAP analysis for interpretability
- Parallel processing with joblib

## Models

**Algorithms:**
- Random Forest (RF)
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Multi-Layer Perceptron (MLP)
- XGBoost (XGB)

**Evaluation Metrics:**
- Accuracy, F1-score (weighted)
- Precision, recall per class
- Confusion matrices
- Cross-validation scores

## Results Structure

```
Resultados/
├── classification_exclude_prod/              # NPK models without productivity vars
│   ├── most_frequent_variables_70.json       # Top 70% important features
│   ├── most_frequent_variables_80.json       # Top 80% important features
│   ├── resultados_modelos_completos.csv      # Complete model metrics
│   └── RF/SVM/KNN/MLP/XGB/                   # Per-model results
├── classification_cuartiles_exclude_prod/    # Quartile classification
├── classification_cuartiles_less_vars/       # Reduced features
├── classification_cuartiles_pca/             # PCA-transformed
├── permutation_importance/                   # Permutation importance results
└── treatments_quantile_unified.json          # Treatment quantile definitions
```

## Key Findings

**Most Important Variables:**
Consistent across models (SHAP + permutation importance):
- Soil nutrients: K_suelo_Horiba, Ca_suelo_Horiba, NO3_suelo_Horiba
- Sap nutrients: K_savia, Ca_savia, Conductividad_savia
- Plant health: Clorofila (SPAD)
- Soil properties: VWC (water content), EC, pH

**Model Performance:**
- XGBoost and Random Forest: Best overall performance
- Quartiles Models trained with reduced features (top 80%) maintain similar accuracy
- Quartile classification more stable than fertilizer classification

## Workflow

1. **Data Integration**: Run notebooks 2-preprocess and 3-Preprocess
2. **Generate Imputed Data**: Complete 3-Preprocess notebook to create `df_imputed_corrected.csv`
3. **Train Models**: Execute train scripts (NPK or quartiles)
4. **Feature Selection**: Extract top variables from results
5. **Retrain**: Use reduced variable set for optimized models
6. **Interpret**: Analyze SHAP values and permutation importance
7. **Validate**: Review productivity correlations

## Requirements

```
pandas, numpy, scikit-learn
xgboost, imbalanced-learn
shap, eli5
matplotlib, seaborn, plotly
```

## Usage

```bash
# Train individual NPK models
python train/train_NPK.py

# Train quartile classification
python train/train_cuartiles.py

# Train with reduced variables
python train/train_cuartiles_less_vars.py
```

