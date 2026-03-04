"""
Utilities for merging and combining results from different model types.
This module provides functions to merge results from traditional ML models
and transformer-based models.
"""

import pickle
import pandas as pd
import numpy as np
import copy


def merge_model_results(results_dict1, results_dict2, verify_structure=True):
    """
    Une dos diccionarios de resultados de modelos.
    
    Args:
        results_dict1 (dict): Primer diccionario de resultados (ej: modelos tradicionales).
                             Formato: {model_name: [lista de resultados]}
        results_dict2 (dict): Segundo diccionario de resultados (ej: modelos transformer).
                             Formato: {model_name: [lista de resultados]}
        verify_structure (bool): Si True, verifica que la estructura sea consistente.
    
    Returns:
        dict: Diccionario combinado con todos los resultados.
              Formato: {model_name: [lista de resultados]}
    
    Example:
        >>> traditional_results = {'RF': [...], 'SVM': [...]}
        >>> transformer_results = {'SwiFT': [...], 'TabNet': [...]}
        >>> all_results = merge_model_results(traditional_results, transformer_results)
        >>> print(all_results.keys())
        dict_keys(['RF', 'SVM', 'SwiFT', 'TabNet'])
    """
    # Crear una copia profunda para no modificar los originales
    merged_results = copy.deepcopy(results_dict1)
    
    # Verificar que no haya modelos duplicados
    common_keys = set(results_dict1.keys()) & set(results_dict2.keys())
    if common_keys:
        print(f"Warning: Found duplicate model names: {common_keys}")
        print("Models from second dictionary will overwrite those from first dictionary.")
    
    # Añadir los resultados del segundo diccionario
    for model_name, results_list in results_dict2.items():
        merged_results[model_name] = results_list
    
    # Verificar estructura si se solicita
    if verify_structure:
        verify_results_structure(merged_results)
    
    print(f"\nMerged {len(results_dict1)} models from first dict and {len(results_dict2)} models from second dict.")
    print(f"Total models in merged dict: {len(merged_results)}")
    print(f"Models: {list(merged_results.keys())}")
    
    return merged_results


def verify_results_structure(results_dict):
    """
    Verifica que la estructura del diccionario de resultados sea consistente.
    
    Args:
        results_dict (dict): Diccionario de resultados a verificar.
    
    Returns:
        bool: True si la estructura es válida, False en caso contrario.
    """
    print("\nVerifying results structure...")
    
    required_keys = ['n_clases', 'model_name', 'accuracy_train', 'accuracy_test',
                     'f1_train', 'f1_test', 'best_params', 'class_distribution']
    
    all_valid = True
    
    for model_name, results_list in results_dict.items():
        if not isinstance(results_list, list):
            print(f"  ❌ {model_name}: Results must be a list, got {type(results_list)}")
            all_valid = False
            continue
        
        for i, result in enumerate(results_list):
            if not isinstance(result, dict):
                print(f"  ❌ {model_name}[{i}]: Result must be a dict, got {type(result)}")
                all_valid = False
                continue
            
            missing_keys = [key for key in required_keys if key not in result]
            if missing_keys:
                print(f" WARNING: {model_name}[{i}]: Missing keys: {missing_keys}")
                all_valid = False
    
    if all_valid:
        print(" All results have valid structure")
    
    return all_valid


def load_results_from_pickle(pickle_path):
    """
    Carga resultados desde un archivo pickle.
    
    Args:
        pickle_path (str): Ruta al archivo pickle.
    
    Returns:
        dict: Diccionario de resultados.
    """
    try:
        with open(pickle_path, 'rb') as f:
            results = pickle.load(f)
        print(f"Successfully loaded results from: {pickle_path}")
        print(f"Models found: {list(results.keys())}")
        return results
    except FileNotFoundError:
        print(f"Error: File not found: {pickle_path}")
        return {}
    except Exception as e:
        print(f"Error loading pickle file: {str(e)}")
        return {}


def save_merged_results(merged_results, output_path):
    """
    Guarda los resultados combinados en un archivo pickle.
    
    Args:
        merged_results (dict): Diccionario de resultados combinados.
        output_path (str): Ruta donde guardar el archivo.
    
    Returns:
        bool: True si se guardó exitosamente, False en caso contrario.
    """
    try:
        with open(output_path, 'wb') as f:
            pickle.dump(merged_results, f)
        print(f"\nMerged results saved successfully to: {output_path}")
        return True
    except Exception as e:
        print(f"Error saving merged results: {str(e)}")
        return False


def create_comparison_dataframe(merged_results):
    """
    Crea un DataFrame con métricas comparativas de todos los modelos.
    
    Args:
        merged_results (dict): Diccionario de resultados combinados.
    
    Returns:
        pd.DataFrame: DataFrame con métricas comparativas.
    """
    rows = []
    
    for model_name, results_list in merged_results.items():
        for result in results_list:
            row = {
                'Model': model_name,
                'n_clases': result.get('n_clases', 'N/A'),
                'Accuracy_Train': result.get('accuracy_train', np.nan),
                'Accuracy_Test': result.get('accuracy_test', np.nan),
                'F1_Train': result.get('f1_train', np.nan),
                'F1_Test': result.get('f1_test', np.nan),
            }
            
            # Añadir información sobre la distribución de clases si está disponible
            if 'class_distribution' in result:
                class_dist = result['class_distribution']
                if hasattr(class_dist, 'to_dict'):
                    row['Class_Distribution'] = str(class_dist.to_dict())
                else:
                    row['Class_Distribution'] = str(class_dist)
            
            rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Ordenar por F1_Test descendente
    if 'F1_Test' in df.columns:
        df = df.sort_values('F1_Test', ascending=False)
    
    return df


def print_comparison_summary(merged_results):
    """
    Imprime un resumen comparativo de todos los modelos.
    
    Args:
        merged_results (dict): Diccionario de resultados combinados.
    """
    print("\n" + "="*80)
    print("COMPARISON SUMMARY - ALL MODELS")
    print("="*80)
    
    df = create_comparison_dataframe(merged_results)
    
    print("\nTop 10 Models by F1 Score (Test):")
    print(df[['Model', 'n_clases', 'F1_Test', 'Accuracy_Test']].head(10).to_string(index=False))
    
    print("\n" + "-"*80)
    print("Average Performance by Model Type:")
    print("-"*80)
    
    avg_by_model = df.groupby('Model')[['Accuracy_Test', 'F1_Test']].mean()
    avg_by_model = avg_by_model.sort_values('F1_Test', ascending=False)
    print(avg_by_model.to_string())
    
    print("\n" + "="*80)


def merge_and_save_all(traditional_path, transformer_path, output_path, 
                       create_summary=True):
    """
    Función completa para cargar, combinar y guardar resultados de ambos tipos de modelos.
    
    Args:
        traditional_path (str): Ruta al pickle con resultados de modelos tradicionales.
        transformer_path (str): Ruta al pickle con resultados de modelos transformer.
        output_path (str): Ruta donde guardar los resultados combinados.
        create_summary (bool): Si True, crea y muestra un resumen comparativo.
    
    Returns:
        dict: Diccionario de resultados combinados.
    
    Example:
        >>> merged = merge_and_save_all(
        ...     'results/traditional_models.pkl',
        ...     'results/transformer_models.pkl',
        ...     'results/all_models_combined.pkl'
        ... )
    """
    print("\n" + "="*80)
    print("MERGING MODEL RESULTS")
    print("="*80)
    
    # Cargar resultados
    print("\n1. Loading traditional models...")
    traditional_results = load_results_from_pickle(traditional_path)
    
    print("\n2. Loading transformer models...")
    transformer_results = load_results_from_pickle(transformer_path)
    
    if not traditional_results and not transformer_results:
        print("\nError: No results to merge!")
        return {}
    
    # Combinar resultados
    print("\n3. Merging results...")
    merged_results = merge_model_results(traditional_results, transformer_results)
    
    # Guardar resultados combinados
    print("\n4. Saving merged results...")
    save_merged_results(merged_results, output_path)
    
    # Crear resumen si se solicita
    if create_summary:
        print_comparison_summary(merged_results)
    
    return merged_results


# Example usage:
if __name__ == "__main__":
    """
    Example of how to use these functions:
    
    # Load and merge results
    traditional_path = '../Resultados/classification_exclude_prod/class_results_individual_elements.pkl'
    transformer_path = '../Resultados/classification_exclude_prod_transformer/class_results_individual_elements_transformer.pkl'
    output_path = '../Resultados/classification_exclude_prod/class_results_all_models_combined.pkl'
    
    merged_results = merge_and_save_all(traditional_path, transformer_path, output_path)
    
    # Or do it manually:
    traditional = load_results_from_pickle(traditional_path)
    transformers = load_results_from_pickle(transformer_path)
    merged = merge_model_results(traditional, transformers)
    save_merged_results(merged, output_path)
    print_comparison_summary(merged)
    """
    pass
