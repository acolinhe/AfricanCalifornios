import os
import csv
import sys
import json
import pandas as pd
from mappings import mappings  # Import mappings from a separate file


def fetch_local_tsv(file_path):
    """
    Reads a local TSV file and returns its contents as a string.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def parse_tsv_data(tsv_data):
    """
    Parses TSV data into a list of dictionaries.
    """
    reader = csv.DictReader(tsv_data.splitlines(), delimiter='\t')
    return list(reader)


def standardize_column_names(data):
    standardized_data = []
    for row in data:
        print(f"Row before standardization: {row}")
        standardized_row = {key.lower(): value for key, value in row.items() if key is not None}
        print(f"Row after standardization: {standardized_row}")
        standardized_data.append(standardized_row)
    return standardized_data



def transformData(data, mapping):
    """
    Transforms the data using the mapping and combines first/last names into a single field.
    """
    transformedData = []
    for row in data:
        def combine_fields(fields):
            # Combine fields if a list is provided, skip None values
            if isinstance(fields, list):
                return ' '.join(row.get(field, '').strip() for field in fields if field)
            return row.get(fields, None)

        transformedRow = {
            'name': combine_fields(mapping.get('name')),
            'race': combine_fields(mapping.get('race')),
            'gender': combine_fields(mapping.get('gender')),
            'father': combine_fields(mapping.get('father')),
            'mother': combine_fields(mapping.get('mother')),
            'spouse': combine_fields(mapping.get('spouse'))
        }
        transformedData.append(transformedRow)
    return transformedData



def write_json_data(data, output_file):
    """
    Writes transformed data to a JSON file.
    """
    with open(output_file, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False)


def main():
    """
    Main function to process all TSV files for the given dataset keys.
    """
    dataset_files = {
        "1790_census": "1790 Census Data Complete.tsv",
        "baptisms": "Baptisms.tsv",
        "padron_1767": "padron_267.tsv",
        "padron_1781": "padron_1781.tsv",
        "padron_1785": "padron_1785.tsv",
        "padron_1821": "padron_1821.tsv"
    }

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))

    for dataset_key, file_name in dataset_files.items():
        print(f"Processing {dataset_key}...")
        file_path = os.path.join(base_dir, file_name)

        # Check if mapping exists for the dataset
        mapping = mappings.get(dataset_key)
        if not mapping:
            print(f"Error: No mapping found for dataset key '{dataset_key}'")
            continue

        try:
            tsv_data = fetch_local_tsv(file_path)
            parsed_data = parse_tsv_data(tsv_data)
            standardized_data = standardize_column_names(parsed_data)
            transformed_data = transformData(standardized_data, mapping)

            # Write output JSON file
            output_file = f"{dataset_key}_output.json"
            write_json_data(transformed_data, output_file)
            print(f"Transformed data saved to {output_file}")
        except Exception as e:
            print(f"Error processing {dataset_key}: {e}")


if __name__ == "__main__":
    main()
