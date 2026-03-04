"""
Script para unir resultados de modelos tradicionales y transformers.
Este script combina los resultados de ambos tipos de modelos en un solo diccionario.
"""

from merge_results_utils import merge_and_save_all, load_results_from_pickle, print_comparison_summary
from config import CFG
import os


def main():
    """
    Función principal para unir resultados de modelos tradicionales y transformers.
    """
    print("\n" + "="*80)
    print("MERGING TRADITIONAL AND TRANSFORMER MODEL RESULTS")
    print("="*80)
    
    # Definir rutas
    # Ruta de modelos tradicionales (RF, SVM, KNN, MLP, XGB)
    traditional_path = f'{CFG.Root}/Resultados/classification_exclude_prod/class_results_individual_elements.pkl'
    
    # Ruta de modelos transformer (SwiFT, TTL, TabNet)
    transformer_path = f'{CFG.Root}/Resultados/classification_exclude_prod_transformer/class_results_individual_elements_transformer.pkl'
    
    # Ruta de salida para resultados combinados
    output_path = f'{CFG.Root}/Resultados/classification_exclude_prod/class_results_all_models_combined.pkl'
    
    # Verificar que existan los archivos
    if not os.path.exists(traditional_path):
        print(f"\nWarning: Traditional models file not found: {traditional_path}")
        print("Please run the traditional models training first.")
        return
    
    if not os.path.exists(transformer_path):
        print(f"\nWarning: Transformer models file not found: {transformer_path}")
        print("Please run the transformer models training first (train_NPK_transformers.py).")
        return
    
    # Unir y guardar resultados
    print(f"\nTraditional models path: {traditional_path}")
    print(f"Transformer models path: {transformer_path}")
    print(f"Output path: {output_path}")
    
    merged_results = merge_and_save_all(
        traditional_path=traditional_path,
        transformer_path=transformer_path,
        output_path=output_path,
        create_summary=True
    )
    
    if merged_results:
        print("\n" + "="*80)
        print("✓ MERGE COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"\nCombined results contain {len(merged_results)} model types:")
        for model_name, results_list in merged_results.items():
            print(f"  - {model_name}: {len(results_list)} experiments")
        print(f"\nMerged results saved to: {output_path}")
        print("="*80 + "\n")
    else:
        print("\n❌ Merge failed. Please check the error messages above.")


def merge_with_nested():
    """
    Función para unir resultados de modelos tradicionales nested y transformers.
    """
    print("\n" + "="*80)
    print("MERGING NESTED TRADITIONAL AND TRANSFORMER MODEL RESULTS")
    print("="*80)
    
    # Definir rutas para nested
    traditional_nested_path = f'{CFG.Root}/Resultados/classification_exclude_prod_nested/class_results_individual_elements.pkl'
    transformer_path = f'{CFG.Root}/Resultados/classification_exclude_prod_transformer/class_results_individual_elements_transformer.pkl'
    output_nested_path = f'{CFG.Root}/Resultados/classification_exclude_prod_nested/class_results_all_models_combined.pkl'
    
    # Verificar existencia
    if not os.path.exists(traditional_nested_path):
        print(f"\nWarning: Nested traditional models file not found: {traditional_nested_path}")
        return
    
    if not os.path.exists(transformer_path):
        print(f"\nWarning: Transformer models file not found: {transformer_path}")
        return
    
    # Unir y guardar
    merged_results = merge_and_save_all(
        traditional_path=traditional_nested_path,
        transformer_path=transformer_path,
        output_path=output_nested_path,
        create_summary=True
    )
    
    if merged_results:
        print("\nNESTED MERGE COMPLETED SUCCESSFULLY!")
        print(f"Merged results saved to: {output_nested_path}\n")


def compare_all_approaches():
    """
    Compara los resultados de diferentes enfoques:
    - Modelos tradicionales
    - Modelos tradicionales nested
    - Modelos transformer
    - Modelos combinados
    """
    print("\n" + "="*80)
    print("COMPARING ALL APPROACHES")
    print("="*80)
    
    paths = {
        'Traditional': f'{CFG.Root}/Resultados/classification_exclude_prod/class_results_individual_elements.pkl',
        'Traditional_Nested': f'{CFG.Root}/Resultados/classification_exclude_prod_nested/class_results_individual_elements.pkl',
        'Transformer': f'{CFG.Root}/Resultados/classification_exclude_prod_transformer/class_results_individual_elements_transformer.pkl',
        'Combined': f'{CFG.Root}/Resultados/classification_exclude_prod/class_results_all_models_combined.pkl'
    }
    
    for approach_name, path in paths.items():
        if os.path.exists(path):
            print(f"\n{'-'*60}")
            print(f"Approach: {approach_name}")
            print(f"{'-'*60}")
            results = load_results_from_pickle(path)
            if results:
                print_comparison_summary(results)
        else:
            print(f"\nWarning: {approach_name} results not found: {path}")


if __name__ == "__main__":
    # Opción 1: Unir modelos tradicionales (no nested) con transformers
    main()
    
    # Opción 2: Descomentar para unir modelos nested con transformers
    # merge_with_nested()
    
    # Opción 3: Descomentar para comparar todos los enfoques
    # compare_all_approaches()
