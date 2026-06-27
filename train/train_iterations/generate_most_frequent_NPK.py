import json
from config import CFG
import pandas as pd
from pathlib import Path

# =========== LOAD DATA =================


permutation_path = Path("/home/student/PAI/Analisis-PAI/Resultados/classification_exclude_prod/permutation_importance/most_frequent_variables_TOTAL_80.csv")
#load permutation importance data
permutation_df = pd.read_csv(permutation_path)

shap_path = Path("/home/student/PAI/Analisis-PAI/train/train_iterations/shap_outputs/classification_exclude_prod/common_vars_all_models_100v2.csv")
# load shap values data
shap_df = pd.read_csv(shap_path)

#output path
json_best_variables = f"{CFG.Root}/Resultados/classification_exclude_prod/most_frequent_variables_80.json"

# iterate each column in permutation and shap and save common vars for each column, then save all the unique vars found
shap_df.columns = permutation_df.columns  # align column names for comparison
common_all = {}
for col in permutation_df.columns:
    if col in shap_df.columns:
        perm_vars = set(permutation_df[col].dropna().astype(str))
        shap_vars = set(shap_df[col].dropna().astype(str))
        common_vars = list(perm_vars.intersection(shap_vars))
        common_all[col] = common_vars
        
# save common variables to json
with open(json_best_variables, 'w') as json_file:
    json.dump(common_all, json_file, indent=4)