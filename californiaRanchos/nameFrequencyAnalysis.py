import pandas as pd
import os
from collections import Counter
import matplotlib.pyplot as plt

def extract_names_from_csv(file_path, column_name='Persons (entities)'):
    """Extract names from the specified column of a CSV file."""
    df = pd.read_csv(file_path)
    names = []
    for row in df[column_name].dropna():
        names.extend([name.strip() for name in row.split(',') if name.strip()])
    return names

def save_top_10_names_as_png(frequency_df, output_dir):
    """Save a table of the top 10 names as a PNG image."""
    top_10 = frequency_df.head(10)

    # Plotting the table
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis('off')
    table = ax.table(cellText=top_10.values, colLabels=top_10.columns, cellLoc='center', loc='center')

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.auto_set_column_width(col=list(range(len(top_10.columns))))

    # Save the table as a PNG
    output_path = os.path.join(output_dir, "top_10_names.png")
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Top 10 names table saved as PNG: {output_path}")

def main():
    input_dir = "../data/"
    output_file = os.path.join(input_dir, "name_frequency_table.csv")

    # List of CSV files to process
    csv_files = [
        "landGrantSpacy.csv",
        "landGrantFlair.csv",
        "landGrantHuggingFace.csv",
        "landGrantStanza.csv"
    ]

    all_names = []

    # Extract names from each CSV file
    for csv_file in csv_files:
        file_path = os.path.join(input_dir, csv_file)
        if os.path.exists(file_path):
            print(f"Processing {csv_file}...")
            names = extract_names_from_csv(file_path)
            all_names.extend(names)
        else:
            print(f"File not found: {csv_file}")

    # Count the frequency of each name
    name_counts = Counter(all_names)

    # Create a DataFrame for the frequency table
    frequency_df = pd.DataFrame(name_counts.items(), columns=["Name", "Frequency"])
    frequency_df = frequency_df.sort_values(by="Frequency", ascending=False)

    # Save the frequency table to a CSV file
    frequency_df.to_csv(output_file, index=False)
    print(f"Name frequency table saved to {output_file}")

    # Save the top 10 names as a PNG image
    save_top_10_names_as_png(frequency_df, input_dir)

if __name__ == "__main__":
    main()
