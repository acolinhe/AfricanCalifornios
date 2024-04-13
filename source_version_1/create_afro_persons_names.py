import pandas as pd

terms = [
    "negro", "negra", "mulato", "mulata", "moreno", "morena",
    "color quebrado", "color quebrada", "pardo", "parda",
    "morizco", "morizca", "prieto", "prieta", "triguñeo", "triguñea"
]


def create_afro_1790_census_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path + "/1790 Census Data Complete.csv")
    pattern = '|'.join(terms)

    return df[df['Race'].str.contains(pattern, case=False, na=False)]
