# Execution order

This project follows a simple pipeline for training, summarizing, and explaining models.

## 1. Train models (first)

Run these to generate the training outputs:

- train_cuartiles_iterations.py
- train_NPK_iterations.py

Commands:

```bash
python train_cuartiles_iterations.py
python train_NPK_iterations.py
```

Output folders:
- ../Resultados/ (training artifacts and per-iteration folders)
- ../Resultados/**/permutation_importance/ (created later by XAI step)

## 2. Summarize performance (second)

Run the summary notebooks to aggregate the results and identify best iterations:

- summary_model_performance_cuartiles_iterations.ipynb
- summary_model_performance_iterations.ipynb

Run order:

```text
Open each notebook and run all cells from top to bottom.
```

Output files:
- ../Resultados/*.csv (summary tables saved by the notebooks)

## 3. XAI using best iteration (final)

Use the best iteration from the summary notebooks and run:

- extract_XAI_best_iter.py

Commands:

```bash
python extract_XAI_best_iter.py --model cuartiles --best_iter <ITER>
python extract_XAI_best_iter.py --model npk --best_iter <ITER>
```

Inputs:
- Results from ../Resultados/**/iter_*/resultados_modelos_completos.csv
- Best iteration ID from the summary notebooks

Outputs:
- ./shap_outputs/
- ../Resultados/**/permutation_importance/
