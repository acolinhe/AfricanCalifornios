import pandas as pd
import os

def load_golden_set(file_path):
    """Loads the golden set CSV file."""
    return pd.read_csv(os.path.join(file_path, 'landGrantGoldenSet.csv'))

def create_comparison_dataset(golden_set, ner_set, column_of_interest, model_name):
    """Merges golden set and NER set on 'Grant Number' and prepares the comparison dataset."""
    # Align the rows by 'Grant Number' using a merge
    merged_df = pd.merge(golden_set, ner_set, on='Grant Number', suffixes=('_golden', '_ner'))
    
    # Extract the relevant columns: Grant Number, Golden Names, Generated Names
    comparisons = merged_df[['Grant Number', column_of_interest + '_golden', column_of_interest + '_ner']].copy()
    
    # Rename the columns for clarity
    comparisons.columns = ['Grant Number', 'Golden Names', 'Generated Names']
    
    # Remove '[' and ']' characters from the Golden Names column
    comparisons['Golden Names'] = comparisons['Golden Names'].str.replace('[', '').str.replace(']', '')
    
    # Add the model name as a new column
    comparisons['Model'] = model_name
    
    return comparisons

def process_ner_files(file_path, golden_set, ner_files, column_of_interest):
    """Processes each NER file and merges it with the golden set."""
    all_comparisons_df = pd.DataFrame()  # Initialize an empty dataframe to store all results
    
    for ner_file in ner_files:
        # Load each NER file
        ner_set = pd.read_csv(os.path.join(file_path, ner_file))
        
        # Extract the model name from the file name (for display purposes)
        model_name = ner_file.split('.')[0]  # Use the file name without extension as model name
        
        # Create the comparison dataset for this NER model
        comparison_df = create_comparison_dataset(golden_set, ner_set, column_of_interest, model_name)
        
        # Append the results to the overall dataframe
        all_comparisons_df = pd.concat([all_comparisons_df, comparison_df], ignore_index=True)
    
    return all_comparisons_df

def save_comparisons(output_file, all_comparisons_df):
    """Saves the final comparisons dataframe to a CSV file."""
    all_comparisons_df.to_csv(output_file, index=False)

def main():
    # File path where the datasets are located
    file_path = '../data/'
    
    # Load the golden set
    golden_set = load_golden_set(file_path)
    
    # Focus on the 'Persons (entities)' column
    column_of_interest = 'Persons (entities)'
    
    # List of NER output files to compare with the golden set
    ner_files = [
        'landGrantHuggingFace.csv',
        'landGrantStanza.csv',
        'landGrantSpacy.csv',
    ]
    
    # Process NER files and create a comparison dataset
    all_comparisons_df = process_ner_files(file_path, golden_set, ner_files, column_of_interest)
    
    # Save the combined results to a CSV
    output_file = os.path.join(file_path, "landGrantNERTable.csv")
    save_comparisons(output_file, all_comparisons_df)

if __name__ == "__main__":
    main()
