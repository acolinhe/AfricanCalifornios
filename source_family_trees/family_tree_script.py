import logging
from typing import Optional
import pandas as pd
import os
import json
from family_unit import FamilyUnit, PersonUnit

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def read_data(file_name: str) -> Optional[pd.DataFrame]:
    """ Load data from a CSV file and handle potential errors. """
    current_directory = os.getcwd()
    matches_directory = os.path.join(current_directory, '..', 'people_collect')
    people_collect_path = os.path.join(matches_directory, file_name)

    try:
        return pd.read_csv(people_collect_path)
    except FileNotFoundError:
        logging.error(f"File not found: {people_collect_path}")
        return None
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return None


def add_identifiers(df, id_column_name):
    """ Add sequential identifiers to a dataframe. """
    df[id_column_name] = range(1, len(df) + 1)


# if person has no children, then empty list
def grab_children(row_id, df: pd.DataFrame) -> list:
    row = df.loc[df['ecpp_id'] == row_id]

    child_columns = [col for col in df.columns if col.startswith('Child')]
    children_list = row[child_columns].values.flatten().tolist()
    children_list = [child for child in children_list if pd.notna(child)]

    return children_list


def serialize_family_units_to_json(family_units, filepath):
    family_units_dict = {
        "family_units": [family_unit.to_dict() for family_unit in family_units]
    }

    with open(filepath, 'w') as json_file:
        json.dump(family_units_dict, json_file, indent=4)


# working with pc2, baptisms, and 1790 census for now
# need to add spouse and go down and up family trees (ask dr. Jones!)
def main():
    pid = 101
    # put in main for now and abstract away with functions
    census_1790 = read_data('1790 Census Data Complete.csv')
    baptisms = read_data('Baptisms.csv')
    add_identifiers(census_1790, 'ecpp_id')
    add_identifiers(baptisms, '#ID')
    pc2 = read_data('people_collect_2.csv')
    family_units = []

    for index, row in pc2.iterrows():
        family = FamilyUnit(index)
        person = PersonUnit(pid,
                            name=row['first_name'],
                            sex=row['sex'],
                            race=row['race_aggregated'],
                            ethnicity=row['ethnicity'],
                            baptismal_date=[row['baptismal_date']], )
        pid += 1

        for ch in grab_children(row['ecpp_id'], census_1790):
            child = PersonUnit(pid, name=ch)
            pid += 1
            person.add_potential_child(child)

        family.add_member(person)
        family_units.append(family)

    serialize_family_units_to_json(family_units, 'family_units.json')


if __name__ == '__main__':
    main()
