from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from utils import *


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract SHAP feature importance for the best iteration and summarize results."
        )
    )
    parser.add_argument(
        "--best-iter",
        type=int, default=13, help="Best iteration index (default: 13)",
    )
    parser.add_argument(
        "--seed-offset",
        type=int, default=41, help="Seed offset used to build the model path (default: 41)",
    )
    parser.add_argument(
        "--path-model",
        type=str, default="",
        help=(
            "Full path to the model .pkl file. If empty, it is built from CFG.Root, "
            "--results-subdir, --best-iter, and --seed-offset."
        ),
    )
    parser.add_argument(
        "--results-subdir",
        type=str, default="Resultados/classification_exclude_prod",
        help="Subdir under CFG.Root for model results.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,  help="Number of SHAP iterations to run (default: 10)",
    )
    parser.add_argument(
        "--common-threshold",
        type=float,  default=80.0, help="Minimum percent of columns for frequent features (default: 80)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",  help=(
            "Output subdir under shap_outputs. Defaults to the model name derived "
            "from the model path."
        ),
    )
    parser.add_argument(
        "--element",
        type=str, default="Nitrogen",  help="Target element for data preparation (default: Nitrogen)",
    )
    return parser.parse_args()


def build_model_path(cfg_root: str, args: argparse.Namespace) -> str:
    if args.path_model:
        return args.path_model

    iter_seed = args.best_iter + args.seed_offset
    model_dir = Path(cfg_root) / args.results_subdir / f"iter_{args.best_iter}_seed_{iter_seed}"
    return str(model_dir / "class_results_individual_elements.pkl")


def get_model_name_from_path(path_model: str) -> str:
    # Two levels up: .../<model_name>/iter_x_seed_y/class_results_...
    return Path(path_model).resolve().parents[1].name


def load_results(path_model: str) -> dict:
    if not path_model:
        raise ValueError("Model path is empty.")

    with open(path_model, "rb") as pkl_file:
        return pickle.load(pkl_file)


def infer_training_mode(model_dir_name: str) -> None:
    if "cuartiles" in model_dir_name:
        CFG.cuartiles_train = True
        CFG.individual_train = False
    else:
        CFG.cuartiles_train = False
        CFG.individual_train = True


def prepare_data(best_iter: int, element: str) -> tuple:
    df_imputed = pd.read_csv(CFG.path_df_imputed_corrected)
    df_imputed.columns = clean_feature_names(df_imputed.columns)

    n_clases = 2 if CFG.cuartiles_train else 3

    return preparar_datos(
        df_imputed,
        n_clases=n_clases,
        element=element,
        best_variables=None,
        CFG=CFG,
        random_state=best_iter + 41,
    )


def plot_shap_importance(
    model,
    X_test,
    feature_names,
    model_type="tree",
    n_clases=2,
    title="SHAP Feature Importance",
    path=None,
    iteration=0,
):
    """Genera graficos de importancia SHAP."""
    X_df = pd.DataFrame(X_test, columns=feature_names)

    if model_type == "tree":
        try:
            if hasattr(model, "get_booster"):
                explainer = shap.TreeExplainer(model, model_output="raw")
            else:
                explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_df)
        except (ValueError, AttributeError) as exc:
            if "could not convert string to float" in str(exc) or "base_score" in str(exc):
                print(
                    f"Warning: TreeExplainer failed for {type(model).__name__}. "
                    "Using KernelExplainer as fallback."
                )
                background = shap.sample(X_df, min(100, len(X_df)), random_state=42 + iteration)
                explainer = shap.KernelExplainer(model.predict_proba, background)
                shap_values = explainer.shap_values(background, nsamples=100)
                X_df_used = background

                fig = plt.figure(figsize=(12, 6))
                shap.summary_plot(
                    shap_values,
                    X_df_used,
                    feature_names=feature_names,
                    plot_type="bar",
                    show=False,
                )
                plt.title(title, fontsize=14, pad=20)
                plt.xlabel("Mean |SHAP value|", fontsize=12)
                if path:
                    plt.savefig(f"{path}_bar_{iteration}.png", dpi=300, bbox_inches="tight")
                plt.tight_layout()
                return shap_values, explainer, fig, X_df_used
            raise

        X_df_used = X_df
    else:
        background = shap.sample(X_df, min(100, len(X_df)), random_state=42 + iteration)
        explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_values = explainer.shap_values(background, nsamples=100)
        X_df_used = background

    fig = plt.figure(figsize=(12, 6))
    shap.summary_plot(
        shap_values,
        X_df_used,
        feature_names=feature_names,
        plot_type="bar",
        show=False,
    )
    plt.title(title, fontsize=14, pad=20)
    plt.xlabel("Mean |SHAP value|", fontsize=12)
    if path:
        plt.savefig(f"{path}_bar_{iteration}.png", dpi=300, bbox_inches="tight")
    plt.tight_layout()

    return shap_values, explainer, fig, X_df_used


def extract_common_values_from_csv(csv_path: str) -> list:
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

    common_values = set.intersection(*column_sets)

    if not common_values:
        return []

    scores = {}
    for val in common_values:
        scores[val] = sum(rank_dict[val] for rank_dict in column_rankings)

    sorted_values = sorted(scores.keys(), key=lambda v: scores[v])

    return sorted_values


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


def save_iteration_results(results_iteration_shap: dict, output_dir: str) -> None:
    output_path = os.path.join(output_dir, "results_iteration_shap.json")
    with open(output_path, "w") as file_obj:
        json.dump(results_iteration_shap, file_obj, indent=4)


def save_algorithm_csvs(results_iteration_shap: dict, output_dir: str) -> None:
    models = results_iteration_shap["5"].keys()

    for model_name in models:
        model_dir = os.path.join(output_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)

        for iteration in results_iteration_shap.keys():
            algs = results_iteration_shap[iteration][model_name].keys()
            for alg_name in algs:
                features = results_iteration_shap[iteration][model_name][alg_name]
                df_features = pd.DataFrame({f"Iteration_{int(iteration) + 1}": features})
                csv_path = os.path.join(model_dir, f"vars_{alg_name}.csv")
                if os.path.exists(csv_path):
                    df_existing = pd.read_csv(csv_path)
                    df_combined = pd.concat([df_existing, df_features], axis=1)
                    df_combined.to_csv(csv_path, index=False)
                else:
                    df_features.to_csv(csv_path, index=False)


def save_common_vars(output_dir: str, threshold_percentage: float) -> dict:
    common_all_models = {}

    for model in os.listdir(output_dir):
        model_dir = os.path.join(output_dir, model)
        if not os.path.isdir(model_dir):
            continue

        csv_files = [f for f in os.listdir(model_dir) if f.endswith(".csv")]
        if not csv_files:
            continue

        common_vars_dict = {}

        for csv_file in csv_files:
            csv_path = os.path.join(model_dir, csv_file)
            if "common" in csv_path:
                continue

            common_vars = extract_frequent_values_from_csv(
                csv_path, threshold_percentage=threshold_percentage
            )
            algorithm_name = csv_file.replace("vars_", "").replace(".csv", "")
            common_vars_dict[algorithm_name] = common_vars

        if common_vars_dict:
            df_common = pd.DataFrame.from_dict(common_vars_dict, orient="index").T
            output_csv = os.path.join(
                model_dir, f"common_vars_{model}_{int(threshold_percentage)}.csv"
            )
            df_common.to_csv(output_csv, index=False)
            print(f"\nGuardado: {output_csv}")

        common_csv_path = os.path.join(
            model_dir, f"common_vars_{model}_{int(threshold_percentage)}.csv"
        )
        if os.path.exists(common_csv_path):
            common_vars = extract_frequent_values_from_csv(
                common_csv_path, threshold_percentage=100
            )
            common_all_models[model] = common_vars

    df_common_all_models = pd.DataFrame.from_dict(common_all_models, orient="index").T
    df_common_all_models.to_csv(
        os.path.join(output_dir, "common_vars_all_models_100v2.csv"), index=False
    )

    return common_all_models


def run_shap_pipeline(
    all_results: dict,
    feature_names: list,
    X_test,
    model_name: str,
    iterations: int,
) -> dict:
    results_iteration_shap = {}

    for i in range(1, iterations + 1):
        for algorithm, value in all_results.items():
            counter = 0
            for modelo in value:
                print(f"Iteracion {i + 1} - Algoritmo: {algorithm}")
                if "grid_search" in modelo:
                    pipeline = modelo["grid_search"].best_estimator_
                else:
                    pipeline = modelo["best_model"]

                output_dir = os.path.join(os.getcwd(), "shap_outputs", model_name, str(i))
                os.makedirs(output_dir, exist_ok=True)

                scaler = pipeline.named_steps["scaler"]
                clf = pipeline.named_steps["clf"]
                X_test_scaled = scaler.transform(X_test)

                shap_values, explainer, fig, X_df_used = plot_shap_importance(
                    clf,
                    X_test_scaled,
                    feature_names,
                    model_type=MODELS_CONFIG[algorithm]["model_type"],
                    n_clases=2,
                    title=f"SHAP Feature Importance - {algorithm} Iteration {i + 1}",
                    path=os.path.join(
                        output_dir, f"shap_importance_{algorithm}_iteration_{i + 1}"
                    ),
                    iteration=i,
                )
                plt.close(fig)

                all_results[algorithm][counter]["shap_values"] = shap_values
                all_results[algorithm][counter]["explainer"] = explainer
                all_results[algorithm][counter]["X_scaled_df"] = X_df_used.copy()
                counter += 1

        best_x_percentage_all_algorithms = extract_top_x_percent_features(
            all_results, percent=0.8, class_path=output_dir, CFG=CFG
        )
        results_iteration_shap[i] = best_x_percentage_all_algorithms

    return results_iteration_shap


def main() -> None:
    args = parse_args()

    path_model = build_model_path(CFG.Root, args)
    all_results = load_results(path_model)

    model_name = get_model_name_from_path(path_model)
    infer_training_mode(model_name)

    output_dir_name = args.output_dir.strip() or model_name

    print(f"Archivo cargado: {path_model}")
    print(f"Nombre del modelo: {model_name}")
    print(f"Entrenamiento por cuartiles: {CFG.cuartiles_train}")
    print(f"Entrenamiento individual: {CFG.individual_train}")

    _, X_test, _, _, feature_names, _ = prepare_data(args.best_iter, args.element)

    results_iteration_shap = run_shap_pipeline(
        all_results,
        feature_names,
        X_test,
        output_dir_name,
        args.iterations,
    )

    output_dir = os.path.join(os.getcwd(), "shap_outputs", output_dir_name)
    save_iteration_results(results_iteration_shap, output_dir)

    with open(os.path.join(output_dir, "results_iteration_shap.json"), "r") as f:
        results_iteration_shap = json.load(f)

    save_algorithm_csvs(results_iteration_shap, output_dir)
    save_common_vars(output_dir, args.common_threshold)


if __name__ == "__main__":
    main()

# python extract_XAI_best_iter_refactored.py \
#   --best-iter 1 \ 
#   --path-model /home/student/PAI/Analisis-PAI/Resultados/classification_cuartiles_exclude_prod/iter_01_seed_42/class_models_cuartiles_all_models.pkl
#   --output-dir cuartiles_iter1