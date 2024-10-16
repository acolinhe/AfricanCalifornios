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
    
    # Remove '[' and ']' characters from the Golden Names and Generated Names columns
    comparisons['Golden Names'] = comparisons['Golden Names'].str.replace('[', '').str.replace(']', '')
    comparisons['Generated Names'] = comparisons['Generated Names'].str.replace('[', '').str.replace(']', '').str.replace("'", "")
    
    # Add the model name as a new column
    comparisons['Model'] = model_name
    
    # Calculate 'Match' by checking if each name in Generated Names is a substring of Golden Names
    comparisons['Match'] = comparisons.apply(lambda row: count_substring_matches(row['Generated Names'], row['Golden Names']), axis=1)
    
    # Calculate 'Match Percentage'
    comparisons['Match Percentage'] = comparisons.apply(lambda row: calculate_match_percentage(row['Match'], row['Generated Names']), axis=1)
    
    return comparisons

def count_substring_matches(generated_names, golden_names):
    """Counts how many names from 'Generated Names' are valid substrings of 'Golden Names'
    and at least as long as the matched part in 'Golden Names'."""
    if pd.isna(generated_names) or pd.isna(golden_names):
        return 0
    
    # Split names by commas and trim spaces
    generated_list = [name.strip() for name in generated_names.split(',')]
    
    # Count how many names in generated_list are substrings of the entire golden_names string,
    # and also check that the length of the name in Generated Names is at least as long as
    # the substring match in Golden Names.
    matches = 0
    for gen_name in generated_list:
        if gen_name in golden_names:
            # Check the length of the match in the Golden Names
            golden_match_len = len(golden_names[golden_names.index(gen_name):golden_names.index(gen_name) + len(gen_name)])
            if len(gen_name) >= golden_match_len:
                matches += 1
                
    return matches


def calculate_match_percentage(matches, generated_names):
    """Calculates the match percentage (number of matches over total names in Generated Names) and rounds it to two decimal places."""
    if pd.isna(generated_names) or matches == 0:
        return 0.0
    
    # Split generated names by commas and count them
    generated_list = [name.strip() for name in generated_names.split(',')]
    total_names = len(generated_list)
    
    # Calculate percentage and round to 2 decimal places
    return round((matches / total_names) * 100, 0)

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

def calculate_average_hit_percentage(comparisons_df):
    """Calculates and prints the average Match Percentage for each NER model."""
    # Group the data by 'Model' and calculate the average 'Match Percentage' for each model
    average_percentages = comparisons_df.groupby('Model')['Match Percentage'].mean()

    # Print the average hit percentage for each model
    print("Average Hit Percentage per Model:")
    for model, avg_percentage in average_percentages.items():
        print(f"Model: {model}, Average Match Percentage: {avg_percentage:.2f}%")


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
        'landGrantFlair.csv'  # Add the new Flair output file here
    ]
    
    # Process NER files and create a comparison dataset
    all_comparisons_df = process_ner_files(file_path, golden_set, ner_files, column_of_interest)
    
    # Save the combined results to a CSV
    output_file = os.path.join(file_path, "landGrantNERTable.csv")
    save_comparisons(output_file, all_comparisons_df)
    
    # Calculate and print the average hit percentage per model
    calculate_average_hit_percentage(all_comparisons_df)

if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
