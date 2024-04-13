from create_afro_persons_names import *
from person_matcher import PersonMatcher
import time


def load_data(file_path):
    """ Load data from a CSV file and handle potential errors. """
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File not found - {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def add_identifiers(df, id_column_name):
    """ Add sequential identifiers to a dataframe. """
    df[id_column_name] = range(1, len(df) + 1)


def clean_names(df):
    """ Clean and combine name fields. """
    df['Ego_Last Name'] = (df['Ego_Paternal Last Name'].fillna('') + ' ' +
                           df['Ego_Maternal Last Name'].fillna('')).str.strip()


def configure_and_match(census, baptisms, config):
    """ Configure the datasets and perform matching. """
    matcher = PersonMatcher(census=census, baptisms=baptisms, config=config)
    matcher.match()
    return matcher.create_matched_records()


if __name__ == '__main__':
    path = '/datasets/acolinhe/data'
    census_config = {
        'ecpp_id_col': 'ecpp_id',
        'records_id_col': '#ID',
        'census': {
            'First Name': 'First',
            'Last Name': 'Last',
            'Gender': 'Gender',
            'Age': 'Age',
        },
        'baptisms': {
            'First Name': 'SpanishName',
            'Last Name': 'Surname',
            'Mother First Name': 'MSpanishName',
            'Mother Last Name': 'MSurname',
            'Father First Name': 'FSpanishName',
            'Father Last Name': 'FSurname',
            'Gender': 'Sex',
            'Age': 'Age',
        }
    }

    afro_1790_census = create_afro_1790_census_df(path)
    baptisms = load_data(path + '/Baptisms.csv')

    if afro_1790_census is not None and baptisms is not None:
        add_identifiers(afro_1790_census, 'ecpp_id')
        add_identifiers(baptisms, '#ID')
        # clean_names(afro_1790_census)
        start_time = time.time()

        matched_results = configure_and_match(afro_1790_census, baptisms, census_config)
        elapsed_time = time.time() - start_time
        print(matched_results)
        print(f"Execution time: {elapsed_time} seconds")
