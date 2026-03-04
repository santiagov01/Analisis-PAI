# CHANGELOG - Transformer Models Implementation

## Version 1.0.0 (Febrero 2026)

### 🎉 Features Added

#### New Files Created
1. **`config_transformers.py`**
   - Configuration for transformer models (SwiFT, TTL, TabNet)
   - Hyperparameter grids for GridSearchCV
   - Quick configuration for fast testing

2. **`merge_results_utils.py`**
   - Functions to merge results from different model types
   - Result structure verification
   - Comparison and summary generation functions

3. **`merge_all_results.py`**
   - Example script to merge results
   - Comparison between traditional and transformer models
   - Multiple merge scenarios supported

4. **`README_TRANSFORMERS.md`**
   - Complete documentation of the transformer system
   - Usage examples and best practices
   - Architecture descriptions

5. **`requirements_transformers.txt`**
   - All required dependencies
   - Version requirements for compatibility

#### Modified Files

1. **`utils.py`**
   - ➕ Added PyTorch and TabNet imports with graceful fallback
   - ➕ Implemented transformer wrapper classes:
     - `TabNetClassifierWrapper`: Sklearn-compatible wrapper for TabNet
     - `SwiFTClassifier` & `SwiFTModel`: Sparse Weighted Fusion Transformer
     - `TTLClassifier` & `TTLModel`: Transformer-based Tabular Learning
   - ➕ New function `train_test_transformers()`: Main training function for transformers
   - ➕ New function `build_transformer_pipeline()`: Pipeline builder for transformers
   - ✅ All transformer classes implement sklearn interface:
     - `fit(X, y)`
     - `predict(X)`
     - `predict_proba(X)`
     - `get_params()` and `set_params()`

2. **`train_NPK_transformers.py`**
   - 🔄 Completely refactored to use transformer models
   - ➕ Imports from `config_transformers`
   - ➕ Sequential training (transformers optimize GPU/CPU internally)
   - ➕ Complete pipeline: training → SHAP → Permutation Importance
   - ✅ Maintains same structure as traditional models for consistency

### 🏗️ Architecture Implementations

#### TabNet
- Attention-based tabular architecture with sparse learning
- Native interpretability with feature importance
- Parameters: `n_d`, `n_a`, `n_steps`, `lambda_sparse`

#### SwiFT (Sparse Weighted Fusion Transformer)
- Transformer with sparse fusion for tabular data
- Efficient memory usage
- Parameters: `d_model`, `nhead`, `num_layers`, `dropout`

#### TTL (Transformer-based Tabular Learning)
- Optimized transformer with Pre-LN architecture
- Better convergence than standard transformers
- Parameters: `d_model`, `nhead`, `num_layers`, `dim_feedforward`

### 📊 Features

#### SHAP Integration
- ✅ Automatic SHAP calculation using KernelExplainer
- ✅ Sample reduction for efficiency (100 points)
- ✅ Consistent with traditional models structure
- ✅ Saves visualization plots automatically

#### Permutation Importance
- ✅ Full integration with existing `permutation_importance_NPK()` function
- ✅ Works seamlessly with transformer models
- ✅ Results saved in same format as traditional models

#### Result Merging
- ✅ `merge_model_results()`: Merge two result dictionaries
- ✅ `verify_results_structure()`: Validate result consistency
- ✅ `create_comparison_dataframe()`: Generate comparison tables
- ✅ `print_comparison_summary()`: Display performance summaries

### 🔧 Technical Details

#### Pipeline Structure
**Traditional Models:**
```
StandardScaler → SMOTE (optional) → Classifier
```

**Transformer Models:**
```
SMOTE (optional) → Classifier (with internal scaling)
```

#### Training Strategy
- **Traditional**: Parallel execution (`n_jobs=-1`)
- **Transformers**: Sequential execution (internal GPU/CPU optimization)

#### SHAP Calculation
- **Traditional**: TreeExplainer or KernelExplainer based on model type
- **Transformers**: Always KernelExplainer (universal compatibility)

### 📁 Directory Structure

New directory hierarchy with `_transformer` suffix:
```
Resultados/
├── classification_exclude_prod/          # Traditional models
└── classification_exclude_prod_transformer/   # Transformer models
    ├── SwiFT/
    ├── TTL/
    ├── TabNet/
    ├── models/
    └── class_results_individual_elements_transformer.pkl
```

### 🎯 Naming Conventions

Maintained consistency across all model types:

**Traditional Models:**
- `'RF'` → Random Forest
- `'SVM'` → Support Vector Machine
- `'KNN'` → K-Nearest Neighbors
- `'MLP'` → Multi-Layer Perceptron
- `'XGB'` → XGBoost

**Transformer Models:**
- `'SwiFT'` → Sparse Weighted Fusion Transformer
- `'TTL'` → Transformer-based Tabular Learning
- `'TabNet'` → TabNet

### ⚙️ Configuration

#### Hyperparameter Grids

**Full Grid (Production):**
- Comprehensive search space
- Longer training time
- Better performance potential

**Quick Grid (Testing):**
- Reduced search space
- Faster execution
- Good for development/testing

Both available in `config_transformers.py`:
- `TRANSFORMERS_CONFIG`: Full grid
- `TRANSFORMERS_CONFIG_QUICK`: Quick grid

### 📝 Result Structure

Results maintain exact same structure as traditional models:
```python
{
    'n_clases': '3_Nitrogen',
    'model_name': 'SwiFT',
    'accuracy_train': float,
    'accuracy_test': float,
    'f1_train': float,
    'f1_test': float,
    'best_params': dict,
    'class_distribution': Series,
    'classification_report': dict,
    'confusion_matrix_test': array,
    'grid_search': GridSearchCV,
    'shap_values': array,
    'X_scaled_df': DataFrame
}
```

### 🔍 Key Improvements

1. **Modularity**: Transformer code is separate but integrates seamlessly
2. **Consistency**: Same structure, naming, and workflow
3. **Flexibility**: Easy to add new transformer architectures
4. **Documentation**: Comprehensive README and inline comments
5. **Error Handling**: Graceful fallbacks for missing dependencies
6. **Compatibility**: Works with existing analysis functions

### ⚠️ Known Limitations

1. **Training Speed**: Transformers are slower than traditional models
2. **Memory Usage**: Higher memory requirements
3. **SHAP Calculation**: Slower due to KernelExplainer requirement
4. **GPU Detection**: Automatic but may require manual configuration

### 🚀 Usage Examples

#### Basic Training
```python
from utils import train_test_transformers
from config_transformers import TRANSFORMERS_CONFIG

resultado = train_test_transformers(
    df_imputed=df_imputed,
    n_clases=None,
    model_name='TabNet',
    model_config=TRANSFORMERS_CONFIG['TabNet'],
    element='Nitrogen',
    CFG=CFG
)
```

#### Merging Results
```python
from merge_results_utils import merge_and_save_all

merged = merge_and_save_all(
    traditional_path='traditional.pkl',
    transformer_path='transformer.pkl',
    output_path='combined.pkl'
)
```

### 🧪 Testing

All functions tested with:
- ✅ Individual element training (Nitrogen, Phosphorus, Potassium)
- ✅ SHAP value calculation
- ✅ Permutation importance
- ✅ Result merging
- ✅ Comparison with traditional models

### 📦 Dependencies

New dependencies added:
- `torch>=1.10.0`
- `pytorch-tabnet>=4.0`

All dependencies listed in `requirements_transformers.txt`

### 🔄 Migration Path

For existing users:
1. Install new dependencies
2. Run `train_NPK_transformers.py` for transformer models
3. Use `merge_all_results.py` to combine with existing results
4. Continue using existing analysis functions

### 📚 Documentation

Added:
- **README_TRANSFORMERS.md**: Complete user guide
- **CHANGELOG.md**: This file
- Inline comments in all new code
- Docstrings for all new functions

### 🎓 Future Enhancements

Potential additions:
- [ ] Additional transformer architectures (FT-Transformer, SAINT)
- [ ] Automatic hyperparameter optimization (Optuna)
- [ ] Model ensemble combinations
- [ ] GPU memory optimization
- [ ] Distributed training support

### 👥 Contributors

- Implementation: GitHub Copilot
- Architecture Design: Based on latest tabular transformer research
- Testing: Integrated with existing codebase

---

## Version History

### 1.0.0 (Febrero 2026)
- Initial release
- SwiFT, TTL, and TabNet implementation
- Complete integration with existing pipeline
- Full documentation

---

**Note**: This implementation maintains backward compatibility with all existing code while adding new transformer capabilities.
