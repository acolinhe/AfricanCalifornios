def add_identifiers(df, id_column_name):
    """ Add sequential identifiers to a dataframe. """
    df[id_column_name] = range(1, len(df) + 1)


def clean_names_padrones(df):
    """ Clean and combine name fields. """
    df['Ego_Last Name'] = (df['Ego_Paternal Last Name'].fillna('') + ' ' +
                           df['Ego_Maternal Last Name'].fillna('')).str.strip()
