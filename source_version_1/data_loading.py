import pandas as pd
import logging


def load_data(file_path):
    """ Load data from a CSV file and handle potential errors. """
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        return None
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return None


def create_afro_df(path: str) -> pd.DataFrame:
    terms = [
        "negro", "negra", "mulato", "mulata", "moreno", "morena",
        "color quebrado", "color quebrada", "pardo", "parda",
        "morizco", "morizca", "prieto", "prieta", "triguñeo", "triguñea"
    ]
    df = pd.read_csv(path)
    pattern = '|'.join(terms)
    return df[df['Race'].str.contains(pattern, case=False, na=False)]
