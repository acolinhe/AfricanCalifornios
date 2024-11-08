import pandas as pd
import os
import matplotlib.pyplot as plt

def load_corpus(file_path):
    """Loads the text corpus and returns the total word count."""
    with open(file_path, 'r') as file:
        text = file.read()
    word_count = len(text.split())
    return word_count

def calculate_total_words(df, column):
    """Calculates the total word count from names in a specified column across all rows in the DataFrame."""
    if column not in df.columns:
        print(f"Warning: Column '{column}' not found in DataFrame. Skipping this file.")
        return 0
    total_words = df[column].apply(lambda x: len(x.split(',')) if pd.notna(x) else 0).sum()
    return total_words

def calculate_correct_matches(df):
    """Calculates the total correct matches from the 'Match' column if it exists."""
    if 'Match' not in df.columns:
        return 0
    return df['Match'].sum()

def calculate_confusion_matrix(df):
    """Calculate True Positives, False Positives, False Negatives, and True Negatives."""
    confusion_matrix_stats = []
    
    for model_name, model_df in df.groupby('Model'):
        TP = model_df['True Positives'].sum()
        FP = model_df['False Positives'].sum()
        FN = model_df['False Negatives'].sum()
        
        total_entities = model_df['Total Names in Generated'].sum() + model_df['Total Names in Golden'].sum()
        TN = total_entities - (TP + FP + FN)
        
        confusion_matrix_stats.append((model_name, TP, FP, FN, TN))
    
    total_TP = df['True Positives'].sum()
    total_FP = df['False Positives'].sum()
    total_FN = df['False Negatives'].sum()
    total_entities = df['Total Names in Generated'].sum() + df['Total Names in Golden'].sum()
    total_TN = total_entities - (total_TP + total_FP + total_FN)
    
    confusion_matrix_stats.append(('Overall', total_TP, total_FP, total_FN, total_TN))
    
    return confusion_matrix_stats


def display_confusion_matrix_table(confusion_matrix_stats):
    """Display confusion matrix as a Matplotlib table."""
    df_confusion = pd.DataFrame(confusion_matrix_stats, columns=["Model", "TP", "FP", "FN", "TN"])
    
    fig, ax = plt.subplots(figsize=(10, len(confusion_matrix_stats) * 0.5 + 1))
    ax.axis('off')
    
    plt.title("Confusion Matrix for NER Models", fontsize=14, weight='bold')
    
    table = ax.table(cellText=df_confusion.values,
                     colLabels=df_confusion.columns,
                     cellLoc='center',
                     loc='center',
                     colColours=["#f2f2f2"] * len(df_confusion.columns))

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)

    plt.show()

def calculate_model_accuracy(comparisons_df, corpus_word_count):
    """Calculates and displays the accuracy of each model and the overall average accuracy."""
    model_stats = []
    total_correct_matches = comparisons_df['Match'].sum()
    total_words_in_generated = comparisons_df['Total Names in Generated'].sum()
    
    for model_name, model_df in comparisons_df.groupby('Model'):
        model_correct_matches = model_df['Match'].sum()
        model_total_words = model_df['Total Names in Generated'].sum()
        
        model_accuracy = round((model_correct_matches / model_total_words) * 100, 2) if model_total_words > 0 else 0
        model_stats.append((model_name, model_correct_matches, model_total_words, model_accuracy))
    
    overall_accuracy = round((total_correct_matches / total_words_in_generated) * 100, 2) if total_words_in_generated > 0 else 0
    model_stats.append(('Overall', total_correct_matches, total_words_in_generated, overall_accuracy))
    
    return model_stats, overall_accuracy


def process_ner_files(file_path, golden_set, ner_files, column_of_interest):
    """Processes each NER file, merges with the golden set, and calculates stats."""
    all_comparisons_df = pd.DataFrame()
    
    for ner_file in ner_files:
        ner_set = pd.read_csv(os.path.join(file_path, ner_file))
        
        model_name = ner_file.split('.')[0]
        
        comparison_df = create_comparison_dataset(golden_set, ner_set, column_of_interest, model_name)
        
        all_comparisons_df = pd.concat([all_comparisons_df, comparison_df], ignore_index=True)
    
    return all_comparisons_df

def create_comparison_dataset(golden_set, ner_set, column_of_interest, model_name):
    """Merges golden set and NER set on 'Grant Number' and prepares the comparison dataset."""
    merged_df = pd.merge(golden_set, ner_set, on='Grant Number', suffixes=('_golden', '_ner'))
    comparisons = merged_df[['Grant Number', column_of_interest + '_golden', column_of_interest + '_ner']].copy()
    comparisons.columns = ['Grant Number', 'Golden Names', 'Generated Names']
    
    comparisons['Golden Names'] = comparisons['Golden Names'].str.replace('[', '').str.replace(']', '')
    comparisons['Generated Names'] = comparisons['Generated Names'].str.replace('[', '').str.replace(']', '').str.replace("'", "")
    
    comparisons['Model'] = model_name
    comparisons['Match'] = comparisons.apply(lambda row: count_substring_matches(row['Generated Names'], row['Golden Names']), axis=1)
    comparisons['Match Percentage'] = comparisons.apply(lambda row: calculate_match_percentage(row['Match'], row['Generated Names']), axis=1)
    comparisons['Total Names in Generated'] = comparisons['Generated Names'].apply(lambda x: len(x.split(',')) if pd.notna(x) else 0)
    comparisons['Total Names in Golden'] = comparisons['Golden Names'].apply(lambda x: len(x.split(',')) if pd.notna(x) else 0)
    
    return comparisons

def count_substring_matches(generated_names, golden_names):
    if pd.isna(generated_names) or pd.isna(golden_names):
        return 0
    generated_list = [name.strip() for name in generated_names.split(',')]
    matches = sum(1 for gen_name in generated_list if gen_name in golden_names)
    return matches

def calculate_match_percentage(matches, generated_names):
    if pd.isna(generated_names) or matches == 0:
        return 0.0
    total_names = len([name.strip() for name in generated_names.split(',')])
    return round((matches / total_names) * 100, 2)

def display_model_statistics(model_stats, overall_accuracy):
    print("Model Statistics:")
    print(f"{'Model':<20} {'Correct Matches':<15} {'Total Words':<12} {'Accuracy (%)':<12}")
    print("-" * 60)
    for model_name, correct_matches, total_words, accuracy in model_stats:
        print(f"{model_name:<20} {correct_matches:<15} {total_words:<12} {accuracy:<12.2f}")
    print(f"\nOverall accuracy across all models: {overall_accuracy:.2f}%")


def display_model_statistics_table(model_stats, overall_accuracy, corpus_word_count):
    df_stats = pd.DataFrame(model_stats, columns=["Model", "Correct Matches", "Total Words", "Accuracy (%)"])
    df_stats.loc[df_stats['Model'] == 'Overall', 'Accuracy (%)'] = overall_accuracy  # Update Overall accuracy if needed

    fig, ax = plt.subplots(figsize=(8, len(model_stats)*0.5 + 1))
    ax.axis('off')
    
    plt.title(f"Total words in corpus: {corpus_word_count}\nModel Statistics", fontsize=14, weight='bold')
    
    table = ax.table(cellText=df_stats.values,
                     colLabels=df_stats.columns,
                     cellLoc='center',
                     loc='center',
                     colColours=["#f2f2f2"] * len(df_stats.columns))

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.2)

    plt.show()

def calculate_type_i_ii_metrics(df):
    """Calculate True Positives, False Positives, and False Negatives for each row, ensuring non-negative values."""
    df['True Positives'] = df['Match']
    
    df['False Positives'] = df['Total Names in Generated'] - df['True Positives']
    
    df['False Negatives'] = df.apply(lambda row: max(0, row['Total Names in Golden'] - row['True Positives']), axis=1)



def main():
    data_path = '../data/'
    corpus_file = os.path.join(data_path, 'selectedGrants.txt')
    golden_set_file = os.path.join(data_path, 'landGrantGoldenSet.csv')
    
    corpus_word_count = load_corpus(corpus_file)
    print(f"Total words in corpus: {corpus_word_count}")
    
    golden_set = pd.read_csv(golden_set_file)
    
    ner_files = [
        'landGrantHuggingFace.csv',
        'landGrantStanza.csv',
        'landGrantSpacy.csv',
        'landGrantFlair.csv'
    ]
    
    column_of_interest = 'Persons (entities)'
    
    all_comparisons_df = process_ner_files(data_path, golden_set, ner_files, column_of_interest)
    
    model_stats, overall_accuracy = calculate_model_accuracy(all_comparisons_df, corpus_word_count)
    
    display_model_statistics_table(model_stats, overall_accuracy, corpus_word_count)
    plt.show()
    
    calculate_type_i_ii_metrics(all_comparisons_df)
    
    confusion_matrix_stats = calculate_confusion_matrix(all_comparisons_df)
    display_confusion_matrix_table(confusion_matrix_stats)
    plt.show()

if __name__ == "__main__":
    main()