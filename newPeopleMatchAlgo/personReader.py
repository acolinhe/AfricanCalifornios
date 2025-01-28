import os
import csv
import pandas as pd
from mappings import mappings  # Import mappings from a separate file

transformed_datasets = {}

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
    """
    Standardizes column names in the data by converting them to lowercase and stripping whitespace.
    """
    return [
        {key.lower().strip(): value for key, value in row.items() if key is not None}
        for row in data
    ]


def safe_convert_to_number(value):
    """
    Safely converts a value to an integer if possible, or returns None for invalid values.
    """
    try:
        return int(value) if value and value.strip().isdigit() else None
    except ValueError:
        return None


def transform_data(data, mapping):
    """
    Transforms the data using the mapping and combines first/last names into a single field.
    """
    transformed_data = []
    for row in data:
        def combine_fields(fields):
            if isinstance(fields, list):
                return ' '.join(row.get(field, '').strip() for field in fields if field)
            return row.get(fields, None)

        transformed_row = {
            'name': combine_fields(mapping.get('name')),
            'race': combine_fields(mapping.get('race')),
            'gender': combine_fields(mapping.get('gender')),
            'father': combine_fields(mapping.get('father')),
            'mother': combine_fields(mapping.get('mother')),
            'spouse': combine_fields(mapping.get('spouse')),
            'age': safe_convert_to_number(row.get(mapping.get('age')))
        }
        transformed_data.append(transformed_row)
    return transformed_data


def load_data():
    """
    Processes all TSV files for the given dataset keys and stores them in memory.
    """
    dataset_files = {
        "1790_census": "1790 Census Data Complete.tsv",
        "baptisms": "Baptisms.tsv",
        "padron_1767": "padron_1767.tsv",
        "padron_1781": "padron_1781.tsv",
        "padron_1785": "padron_1785.tsv",
        "padron_1821": "padron_1821.tsv"
    }

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))

    for dataset_key, file_name in dataset_files.items():
        file_path = os.path.join(base_dir, file_name)

        mapping = mappings.get(dataset_key)
        if not mapping:
            print(f"Error: No mapping found for dataset key '{dataset_key}'")
            continue

        try:
            tsv_data = fetch_local_tsv(file_path)
            parsed_data = parse_tsv_data(tsv_data)
            standardized_data = standardize_column_names(parsed_data)
            transformed_datasets[dataset_key] = transform_data(standardized_data, mapping)
        except Exception as e:
            print(f"Error processing {dataset_key}: {e}")


def get_transformed_data():
    """
    Returns the transformed datasets stored in memory.
    """
    if not transformed_datasets:
        load_data()
    return transformed_datasets

load_data()
