from data_loading import load_data, create_afro_df


def add_identifiers(df, id_column_name):
    """ Add sequential identifiers to a dataframe. """
    df[id_column_name] = range(1, len(df) + 1)


def clean_names_padrones(df):
    """ Clean and combine name fields. """
    df['Ego_Last Name'] = (df['Ego_Paternal Last Name'].fillna('') + ' ' +
                           df['Ego_Maternal Last Name'].fillna('')).str.strip()


def get_config():
    """Matching configuration structure for score matching."""
    return {
        'census_config': {
            'ecpp_id_col': 'ecpp_id',
            'records_id_col': '#ID',
            'census': {'First Name': 'First', 'Last Name': 'Last', 'Gender': 'Gender', 'Age': 'Age'},
            'baptisms': {'First Name': 'SpanishName', 'Last Name': 'Surname', 'Mother First Name': 'MSpanishName',
                         'Mother Last Name': 'MSurname', 'Father First Name': 'FSpanishName',
                         'Father Last Name': 'FSurname', 'Gender': 'Sex', 'Age': 'Age'}
        },
        'padrones_config': {
            'ecpp_id_col': 'ecpp_id',
            'records_id_col': '#ID',
            'census': {'First Name': 'Ego_First Name', 'Last Name': 'Ego_Last Name', 'Gender': 'Sex', 'Age': 'Age'},
            'baptisms': {'First Name': 'SpanishName', 'Last Name': 'Surname', 'Mother First Name': 'MSpanishName',
                         'Mother Last Name': 'MSurname', 'Father First Name': 'FSpanishName',
                         'Father Last Name': 'FSurname', 'Gender': 'Sex', 'Age': 'Age'}
        }
    }


def get_datasets(path: str):
    return {
        'afro_1790_census': create_afro_df(path + "/1790 Census Data Complete.csv"),
        'padron_1781': create_afro_df(path + "/padron_1781.csv"),
        'padron_1785': create_afro_df(path + "/padron_1785.csv"),
        'padron_1821': create_afro_df(path + "/padron_1821.csv"),
        'padron_1778': create_afro_df(path + "/padron_267.csv"),
        'baptisms': load_data(path + '/Baptisms.csv')
    }
