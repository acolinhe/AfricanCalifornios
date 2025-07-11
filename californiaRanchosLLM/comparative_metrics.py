import pandas as pd
import json
import os
from difflib import SequenceMatcher
import re

def normalize_string(s):
    """Normalize strings for comparison"""
    if not s or s == 'nan':
        return ""
    normalized = re.sub(r'\s+', ' ', str(s).strip().lower())

    normalized = re.sub(r'\s+(co\.|county)$', '', normalized)
    return normalized

def normalize_grant_number(grant_num):
    """Normalize grant numbers by removing # prefix"""
    if not grant_num or grant_num == 'nan':
        return ""
    return str(grant_num).strip().lstrip('#')

def extract_values(val):
    """Extract values from various formats (arrays, strings, etc.)"""
    if not val or val == 'nan' or str(val).strip() == '':
        return set()
    
    val_str = str(val).strip()
    

    if isinstance(val, list):
        return set(normalize_string(item) for item in val if item)
    

    if val_str.startswith('[') and val_str.endswith(']'):
        try:

            parsed = json.loads(val_str.replace("'", '"'))
            if isinstance(parsed, list):
                return set(normalize_string(item) for item in parsed if item)
        except:

            items = val_str[1:-1].split(',')
            return set(normalize_string(item.strip(' "\',')) for item in items if item.strip())
    

    if ',' in val_str:
        items = val_str.split(',')
        return set(normalize_string(item) for item in items if item.strip())
    
    # Single value
    return set([normalize_string(val_str)]) if normalize_string(val_str) else set()

def fuzzy_match_score(str1, str2, threshold=0.8):
    """Calculate fuzzy match score between two strings"""
    if not str1 or not str2:
        return 0.0
    return SequenceMatcher(None, str1, str2).ratio()

def compare_values(golden_set, predicted_set):
    """Compare sets with exact matching only"""
    tp = len(golden_set & predicted_set)
    fp = len(predicted_set - golden_set)
    fn = len(golden_set - predicted_set)
    
    return tp, fp, fn

def compare_values_fuzzy(golden_set, predicted_set, fuzzy_threshold=0.8):
    """Compare sets with fuzzy matching - if ANY item from predicted matches ANY item from golden, it's a TP"""
    tp = fp = fn = 0
    matched_golden = set()
    matched_predicted = set()
    
    # Exact matches first
    exact_matches = golden_set & predicted_set
    tp += len(exact_matches)
    matched_golden.update(exact_matches)
    matched_predicted.update(exact_matches)
    

    remaining_golden = golden_set - matched_golden
    remaining_predicted = predicted_set - matched_predicted
    
    for g_val in remaining_golden:
        best_match = None
        best_score = 0
        for p_val in remaining_predicted:
            score = fuzzy_match_score(g_val, p_val)
            if score > best_score and score >= fuzzy_threshold:
                best_score = score
                best_match = p_val
        
        if best_match:
            tp += 1
            matched_predicted.add(best_match)
        else:
            fn += 1
    

    fp = len(predicted_set - matched_predicted)
    
    return tp, fp, fn

def load_golden_set(csv_path):
    df = pd.read_csv(csv_path)
    df = df.fillna('').astype(str)
    

    if 'Grant Number' in df.columns:
        df['Grant Number'] = df['Grant Number'].apply(normalize_grant_number)
    
    return df

def load_llm_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    if isinstance(data, dict):
        data = list(data.values())
    

    data = [record for record in data if record.get('Grant Name') and record.get('Grant Number')]
    
    df = pd.DataFrame(data)
    df = df.fillna('').astype(str)
    

    if 'Grant Number' in df.columns:
        df['Grant Number'] = df['Grant Number'].apply(normalize_grant_number)
    

    column_mappings = {
        'Persons': 'Persons (entities)',
    }
    
    for old_col, new_col in column_mappings.items():
        if old_col in df.columns and old_col != new_col:
            df.rename(columns={old_col: new_col}, inplace=True)
    
    return df


def debug_mismatches(golden_df, llm_df):
    """Debug function to see what's not matching"""
    golden_grants = set(golden_df['Grant Number'].str.strip())
    llm_grants = set(llm_df['Grant Number'].str.strip())
    
    print(f"Golden set grants: {len(golden_grants)}")
    print(f"LLM grants: {len(llm_grants)}")
    print(f"Common grants: {len(golden_grants & llm_grants)}")
    
    only_golden = golden_grants - llm_grants
    only_llm = llm_grants - golden_grants
    
    if only_golden:
        print(f"Only in golden: {sorted(only_golden)}")
    if only_llm:
        print(f"Only in LLM: {sorted(only_llm)}")
    

    sample_grant = "501"  # NIGUEL
    if sample_grant in golden_grants and sample_grant in llm_grants:
        golden_row = golden_df[golden_df['Grant Number'] == sample_grant].iloc[0]
        llm_row = llm_df[llm_df['Grant Number'] == sample_grant].iloc[0]
        
        print(f"\nSample comparison for grant {sample_grant}:")
        print("GOLDEN PERSONS:", golden_row.get('Persons (entities)', ''))
        print("LLM PERSONS:", llm_row.get('Persons (entities)', ''))
        
        golden_persons = extract_values(golden_row.get('Persons (entities)', ''))
        llm_persons = extract_values(llm_row.get('Persons (entities)', ''))
        print("GOLDEN PARSED:", golden_persons)
        print("LLM PARSED:", llm_persons)
        print("INTERSECTION:", golden_persons & llm_persons)

def evaluate_all_columns(golden_csv, llm_json, model_name="LLM", use_fuzzy=True, fuzzy_threshold=0.8):
    golden_df = load_golden_set(golden_csv)
    llm_df = load_llm_json(llm_json)
    

    debug_mismatches(golden_df, llm_df)
    

    print(f"Golden columns: {list(golden_df.columns)}")
    print(f"LLM columns: {list(llm_df.columns)}")
    
    merged = pd.merge(golden_df, llm_df, on='Grant Number', suffixes=('_golden', '_llm'), how='inner')
    print(f"Merged {len(merged)} records")
    
    columns = [col for col in golden_df.columns if col != 'Grant Number']
    
    summary = []
    for col in columns:
        total_tp = total_fp = total_fn = 0
        
        for _, row in merged.iterrows():
            golden_val = row.get(f"{col}_golden", "")
            llm_val = row.get(f"{col}_llm", "")
            
            golden_set = extract_values(golden_val)
            predicted_set = extract_values(llm_val)
            
            if use_fuzzy:
                tp, fp, fn = compare_values_fuzzy(golden_set, predicted_set, fuzzy_threshold)
            else:
                tp, fp, fn = compare_values(golden_set, predicted_set)
            
            total_tp += tp
            total_fp += fp
            total_fn += fn
        

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0
        
        summary.append({
            'Model': model_name,
            'Column': col,
            'Accuracy': round(accuracy, 4),
            'Precision': round(precision, 4),
            'Recall': round(recall, 4),
            'F1': round(f1, 4),
            'TP': total_tp,
            'FP': total_fp,
            'FN': total_fn
        })
        
        print(f"\nModel: {model_name} | Column: {col}")
        print(f"  Accuracy:  {accuracy:.3f}")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall:    {recall:.3f}")
        print(f"  F1 Score:  {f1:.3f}")
        print(f"  TP: {total_tp}, FP: {total_fp}, FN: {total_fn}")
    
    return summary

if __name__ == "__main__":
    golden_csv = "landGrantGoldenSet.csv"
    llm_dir = "."
    llm_files = [
        "03mini.json",
        "ClaudeSonnet3_5.json", 
        "ClaudeSonnet3_7.json",
        "ClaudeSonnet4.json",
        "Gemini2_0Flash.json",
        "Gemini2_5Pro.json", 
        "GPT4_0.json",
        "GPT4_1.json"
    ]
    
    all_results = []
    
    for llm_file in llm_files:
        llm_json = os.path.join(llm_dir, llm_file)
        model_name = llm_file.replace('.json', '')
        
        if os.path.exists(llm_json):
            print(f"\n{'='*50}")
            print(f"=== Evaluating {model_name} ===")
            print(f"{'='*50}")
            
            try:
                model_results = evaluate_all_columns(golden_csv, llm_json, model_name)
                

                if all_results:
                    blank_row = {col: "" for col in ['Model', 'Column', 'Accuracy', 'Precision', 'Recall', 'F1', 'TP', 'FP', 'FN']}
                    all_results.append(blank_row)
                
                all_results.extend(model_results)
                print(f"✓ Successfully evaluated {model_name}")
                
            except Exception as e:
                print(f"✗ Error evaluating {model_name}: {str(e)}")
                continue
                
        else:
            print(f"✗ File not found: {llm_json}")
    

    if all_results:
        combined_df = pd.DataFrame(all_results)
        combined_df.to_csv("all_models_comparison.csv", index=False)
        print(f"\n{'='*50}")
        print(f"All results saved to all_models_comparison.csv")
        print(f"Total models evaluated: {len([r for r in all_results if r.get('Model')])}")
        print(f"{'='*50}")
        

        models_with_data = [r for r in all_results if r.get('Model') and r.get('Column')]
        if models_with_data:
            summary_df = pd.DataFrame(models_with_data)
            print("\nOverall Performance Summary:")
            print(summary_df.groupby('Model')[['Accuracy', 'Precision', 'Recall', 'F1']].mean().round(3))
    else:
        print("No results to save!")