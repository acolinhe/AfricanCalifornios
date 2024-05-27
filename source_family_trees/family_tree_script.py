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


def grab_children(row_id, df: pd.DataFrame) -> list:
    row = df.loc[df['ecpp_id'] == row_id]
    child_columns = [col for col in df.columns if col.startswith('Child')]
    children_list = row[child_columns].values.flatten().tolist()
    children_list = [child for child in children_list if pd.notna(child)]
    return children_list


def grab_spouse(row_id, df: pd.DataFrame, person_sex: Optional[str], pid: int) -> Optional[PersonUnit]:
    row = df.loc[df['ecpp_id'] == row_id]
    spouse_first = row['Spouse_First'].values[0] \
        if 'Spouse_First' in row and pd.notna(row['Spouse_First'].values[0]) else None
    spouse_last = row['Spouse_Last'].values[0] \
        if 'Spouse_Last' in row and pd.notna(row['Spouse_Last'].values[0]) else None

    if spouse_first and spouse_last:
        spouse_name = f"{spouse_first} {spouse_last}"
        spouse_sex = 'F' if person_sex == 'M' else 'M' if person_sex == 'F' else None
        spouse = PersonUnit(pid=pid, name=spouse_name, sex=spouse_sex)
        return spouse
    return None


def grab_parents(row_id, census_df: pd.DataFrame, baptisms_df: pd.DataFrame, pid: int) -> list:
    parents = []

    row = census_df.loc[census_df['ecpp_id'] == row_id]
    if not row.empty:
        father_first = row['Father_First'].values[0] \
            if 'Father_First' in row and pd.notna(row['Father_First'].values[0]) else None
        father_last = row['Father_Last'].values[0] \
            if 'Father_Last' in row and pd.notna(row['Father_Last'].values[0]) else None
        if father_first and father_last:
            father_name = f"{father_first} {father_last}"
            father = PersonUnit(pid=pid, name=father_name, sex='M')
            parents.append(father)
            pid += 1

        mother_first = row['Mother_First'].values[0] \
            if 'Mother_First' in row and pd.notna(row['Mother_First'].values[0]) else None
        mother_last = row['Mother_Last'].values[0] \
            if 'Mother_Last' in row and pd.notna(row['Mother_Last'].values[0]) else None
        if mother_first and mother_last:
            mother_name = f"{mother_first} {mother_last}"
            mother = PersonUnit(pid=pid, name=mother_name, sex='F')
            parents.append(mother)
            pid += 1

    row_baptism = baptisms_df.loc[baptisms_df['#ID'] == row_id]
    if not row_baptism.empty:
        father_first_b = row_baptism['FSpanishName'].values[0] \
            if 'FSpanishName' in row_baptism and pd.notna(row_baptism['FSpanishName'].values[0]) else None
        father_last_b = row_baptism['FSurname'].values[0] \
            if 'FSurname' in row_baptism and pd.notna(row_baptism['FSurname'].values[0]) else None
        father_ethnicity_b = row_baptism['FEthnicity'].values[0] \
            if 'FEthnicity' in row_baptism and pd.notna(row_baptism['FEthnicity'].values[0]) else None
        if father_first_b and father_last_b:
            father_name_b = f"{father_first_b} {father_last_b}"
            father_b = PersonUnit(pid=pid, name=father_name_b, sex='M', ethnicity=father_ethnicity_b)
            parents.append(father_b)
            pid += 1  # Increment PID

        mother_first_b = row_baptism['MSpanishName'].values[0] \
            if 'MSpanishName' in row_baptism and pd.notna(row_baptism['MSpanishName'].values[0]) else None
        mother_last_b = row_baptism['MSurname'].values[0] \
            if 'MSurname' in row_baptism and pd.notna(row_baptism['MSurname'].values[0]) else None
        mother_ethnicity_b = row_baptism['MEthnicity'].values[0] \
            if 'MEthnicity' in row_baptism and pd.notna(row_baptism['MEthnicity'].values[0]) else None
        if mother_first_b and mother_last_b:
            mother_name_b = f"{mother_first_b} {mother_last_b}"
            mother_b = PersonUnit(pid=pid, name=mother_name_b, sex='F', ethnicity=mother_ethnicity_b)
            parents.append(mother_b)
            pid += 1  # Increment PID

    return parents


def serialize_family_units_to_json(family_units, filepath):
    family_units_dict = {
        "family_units": [family_unit.to_dict() for family_unit in family_units]
    }
    with open(filepath, 'w') as json_file:
        json.dump(family_units_dict, json_file, indent=4)


def check_nan(value):
    return None if pd.isna(value) else value


def remove_commas(value):
    return value.replace(',', '') if value else value


def read_and_prepare_data():
    census_1790 = read_data('1790 Census Data Complete.csv')
    baptisms = read_data('Baptisms.csv')
    pc2 = read_data('people_collect_2.csv')

    if census_1790 is not None:
        add_identifiers(census_1790, 'ecpp_id')
    if baptisms is not None:
        add_identifiers(baptisms, '#ID')

    return census_1790, baptisms, pc2


def create_person_unit(pid, row, census_1790, baptisms):
    person = PersonUnit(pid=pid,
                        name=check_nan(row['first_name']) + ' ' + check_nan(row['last_name']),
                        sex=check_nan(row['sex']),
                        race=remove_commas(check_nan(row['race_aggregated'])),
                        ethnicity=check_nan(row['ethnicity']),
                        baptismal_date=check_nan(row['baptismal_date']))

    pid += 1

    spouse = grab_spouse(row['ecpp_id'], census_1790, person.sex, pid)
    if spouse:
        person.set_potential_spouse(spouse)
        pid += 1

    parents = grab_parents(row['ecpp_id'], census_1790, baptisms, pid)
    for parent in parents:
        person.add_potential_parent(parent)
        pid += 1

    return person


def create_family_units(pc2, census_1790, baptisms):
    pid = 101
    family_units = []

    for index, row in pc2.iterrows():
        family = FamilyUnit(index)
        person = create_person_unit(pid, row, census_1790, baptisms)
        pid += 1

        for ch in grab_children(row['ecpp_id'], census_1790):
            child = PersonUnit(pid=pid, name=check_nan(ch))
            pid += 1
            person.add_potential_child(child)

        family.add_member(person)
        family_units.append(family)

    return family_units


def main():
    census_1790, baptisms, pc2 = read_and_prepare_data()

    if pc2 is None or census_1790 is None or baptisms is None:
        logging.error("Data could not be loaded.")
        return

    family_units = create_family_units(pc2, census_1790, baptisms)
    serialize_family_units_to_json(family_units, 'family_units.json')


if __name__ == '__main__':
    main()
