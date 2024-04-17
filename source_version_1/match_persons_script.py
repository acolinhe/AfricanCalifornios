import os
import logging
import pandas as pd
import time
import numpy as np
import re
from multiprocessing import Pool, cpu_count
from data_processing import add_identifiers, clean_names_padrones
from data_loading import load_data, create_afro_df
from person_matcher import PersonMatcher

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def configure_and_match(census, baptisms, config):
    """Configure the datasets and perform matching."""
    matcher = PersonMatcher(census=census, baptisms=baptisms, config=config)
    matcher.match()
    return matcher.matched_records


def parallel_configure_and_match(data_info):
    """Unpack data from the tuple and run the configure_and_match function with detailed logging."""
    census_data, baptisms_data, config, chunk_index = data_info
    start_time = time.time()
    try:
        logging.info(f"Starting processing for chunk {chunk_index}")
        results = configure_and_match(census_data, baptisms_data, config)
        logging.info(f"Completed processing for chunk {chunk_index}")
        logging.info(f"Chunk {chunk_index} processing time: {time.time() - start_time} seconds")
        return results
    except Exception as e:
        logging.error(f"Error processing chunk {chunk_index}: {str(e)}")
        return pd.DataFrame()


def chunk_dataframe(df, num_chunks):
    """Chunk the dataframe into smaller parts, returning chunks along with their indices."""
    chunk_size = len(df) // num_chunks + (len(df) % num_chunks > 0)
    return [(df.iloc[i:i + chunk_size], i) for i in range(0, len(df), chunk_size)]


def load_and_prepare_data(path):
    """Load data and prepare it by adding identifiers and cleaning names, ensuring 'Age' column exists."""
    datasets = {
        'afro_1790_census': create_afro_df(path + "/1790 Census Data Complete.csv"),
        'padron_1781': create_afro_df(path + "/padron_1781.csv"),
        'padron_1785': create_afro_df(path + "/padron_1785.csv"),
        'padron_1821': create_afro_df(path + "/padron_1821.csv"),
        'padron_267': create_afro_df(path + "/padron_267.csv"),
        'baptisms': load_data(path + '/Baptisms.csv')
    }
    for name, dataset in datasets.items():
        if dataset is not None:
            logging.debug(f"Columns in {name}: {dataset.columns}")
            if 'Age' not in dataset.columns:
                logging.warning(f"'Age' column is missing in dataset {name}. Adding default values.")
                dataset['Age'] = np.nan
            add_identifiers(dataset, 'ecpp_id' if 'padron' in name or 'census' in name else '#ID')
            if 'padron' in name:
                clean_names_padrones(dataset)
    return datasets


def save_to_pickle(dataframe, path, filename):
    """Save the dataframe to a pickle file."""
    full_path = os.path.join(path, filename)
    dataframe.to_pickle(full_path)
    logging.info(f"Dataframe saved to {full_path}")


def process_dataset(dataset_key, datasets, config, output_path):
    if dataset_key == 'baptisms' or datasets[dataset_key] is None:
        return

    dataset_start_time = time.time()
    results = parallel_data_processing(datasets[dataset_key], datasets['baptisms'], config, dataset_key)
    threshold = 0.87
    people_collect_2 = create_people_collect_2(results, threshold, datasets['baptisms'], datasets, dataset_key)
    people_collect_2.to_csv(os.path.join(output_path, f'{dataset_key}_people_collect_2.csv'), index=False)
    logging.info(f"{dataset_key} processing time: {time.time() - dataset_start_time} seconds")


def parallel_data_processing(dataset, baptisms, config, dataset_key):
    """Run dataset into max 64 chunks for parallel processing."""
    num_cores = min(64, cpu_count())
    pool = Pool(processes=num_cores)
    try:
        chunks = chunk_dataframe(dataset, num_cores)
        current_config = config['padrones_config'] if 'padron' in dataset_key else config['census_config']
        tasks = [(chunk, baptisms, current_config, index) for chunk, index in chunks]
        results = pool.map(parallel_configure_and_match, tasks)
    finally:
        pool.close()
        pool.join()
    combined_results = pd.concat(results)
    logging.debug(f"Combined results count: {len(combined_results)}")
    return combined_results


def extract_year_from_filename(filename):
    """Extract year from filename assuming the year is always four digits."""
    match = re.search(r'\d{4}', filename)
    if match:
        return match.group(0)
    return None


# Probably overwriting found values, need to figure this out
def create_people_collect_2(matched_results, threshold, baptisms, other_datasets, dataset_key):
    logging.info(f"Starting to process matched results for {dataset_key}.")

    index_key_primary = 'ecpp_id'
    extracted_year = extract_year_from_filename(dataset_key)
    race_year = 'race_' + (extracted_year if extracted_year else 'unknown')
    dataset_primary = other_datasets[dataset_key]
    if '1790' in dataset_key:
        primary_list = ['Race', 'Current_Location', 'Origin Parish', 'Location Other Race']
        data_columns_primary = {
            'direct': primary_list,
            'mother': primary_list,
            'father': primary_list
        }
    else:
        primary_list = ['Race']
        data_columns_primary = {
            'direct': primary_list,
            'mother': primary_list,
            'father': primary_list
        }

    index_key_baptisms = '#ID'
    dataset_baptisms = baptisms
    data_columns_baptisms = {
        'direct': ['SpanishName', 'Surname', 'Ethnicity', 'FmtdDate', 'Mission', 'FSpanishName', 'FSurname',
                   'FMilitaryStatus', 'FOrigin', 'MSpanishName', 'MSurname', 'MOrigin', 'Sex', 'Notes'],
        'mother': ['MSpanishName', 'MSurname', 'MEthnicity', 'FmtdDate', 'Mission', 'Sex', 'Notes'],
        'father': ['FSpanishName', 'FSurname', 'FEthnicity', 'FmtdDate', 'Mission', 'Sex', 'Notes']
    }

    dataset_primary_indexed = dataset_primary.set_index(index_key_primary)
    dataset_baptisms_indexed = dataset_baptisms.set_index(index_key_baptisms)
    logging.info(f"{dataset_key} data and Baptisms data indexed by their respective keys.")

    direct_matches = matched_results[matched_results['Direct_Total_Match_Score'] >= threshold]
    mother_matches = matched_results[matched_results['Mother_Total_Match_Score'] >= threshold]
    father_matches = matched_results[matched_results['Father_Total_Match_Score'] >= threshold]

    direct_data_primary = dataset_primary_indexed.loc[direct_matches[index_key_primary], data_columns_primary['direct']]
    mother_data_primary = dataset_primary_indexed.loc[mother_matches[index_key_primary], data_columns_primary['mother']]
    father_data_primary = dataset_primary_indexed.loc[father_matches[index_key_primary], data_columns_primary['father']]

    direct_data_baptisms = dataset_baptisms_indexed.loc[
        direct_matches[index_key_baptisms], data_columns_baptisms['direct']]
    mother_data_baptisms = dataset_baptisms_indexed.loc[
        mother_matches[index_key_baptisms], data_columns_baptisms['mother']]
    father_data_baptisms = dataset_baptisms_indexed.loc[
        father_matches[index_key_baptisms], data_columns_baptisms['father']]

    combined_primary = pd.concat([direct_data_primary, mother_data_primary, father_data_primary], axis=0)
    combined_baptisms = pd.concat([direct_data_baptisms, mother_data_baptisms, father_data_baptisms], axis=0)

    combined_primary[index_key_primary] = combined_primary.index
    combined_baptisms[index_key_baptisms] = combined_baptisms.index

    combined_primary = combined_primary.reset_index(drop=True)
    combined_baptisms = combined_baptisms.reset_index(drop=True)

    if '1790' in dataset_key:
        combined_primary.rename(columns={'Race': race_year, 'Current_Location': 'location_1790_census',
                                         'Origin Parish': 'origin_parish_1790_census',
                                         'Location Other Race': 'location_other_race'}, inplace=True)
    else:
        combined_primary.rename(columns={'Race': race_year}, inplace=True)

    combined_baptisms['first_name'] = combined_baptisms[['SpanishName', 'MSpanishName', 'FSpanishName']].fillna('').sum(
        axis=1)
    combined_baptisms['last_name'] = combined_baptisms[['Surname', 'MSurname', 'FSurname']].fillna('').sum(axis=1)

    combined_baptisms['ethnicity'] = combined_baptisms[['Ethnicity', 'MEthnicity', 'FEthnicity']].fillna('').sum(axis=1)

    combined_baptisms['baptismal_date'] = combined_baptisms[['FmtdDate']].fillna('').sum(axis=1)

    combined_baptisms['location_ecpp_baptism'] = combined_baptisms[['Mission']].fillna('').sum(axis=1)

    combined_baptisms['father_first_name'] = combined_baptisms[['FSpanishName']].fillna('').sum(axis=1)

    combined_baptisms['father_last_name'] = combined_baptisms[['FSurname']].fillna('').sum(axis=1)

    combined_baptisms['father_military_status'] = combined_baptisms[['FMilitaryStatus']].fillna('').sum(axis=1)

    combined_baptisms['father_origin'] = combined_baptisms[['FOrigin']].fillna('').sum(axis=1)

    combined_baptisms['mother_first_name'] = combined_baptisms[['MSpanishName']].fillna('').sum(axis=1)

    combined_baptisms['mother_last_name'] = combined_baptisms[['MSurname']].fillna('').sum(axis=1)

    combined_baptisms['mother_origin'] = combined_baptisms[['MOrigin']].fillna('').sum(axis=1)

    combined_baptisms['sex'] = combined_baptisms[['Sex']].fillna('').sum(axis=1)

    combined_baptisms['notes'] = combined_baptisms[['Notes']].fillna('').sum(axis=1)

    combined_baptisms.drop(columns=['SpanishName', 'Surname', 'MSpanishName', 'MSurname', 'FSpanishName', 'FSurname',
                                    'Ethnicity', 'MEthnicity', 'FEthnicity', 'FmtdDate', 'Mission', 'FSpanishName',
                                    'FSurname', 'MSpanishName', 'MSurname', 'FMilitaryStatus', 'FOrigin', 'MOrigin',
                                    'Sex', 'Notes'],
                           inplace=True)

    final_output = pd.concat([combined_primary, combined_baptisms], axis=1)

    desired_order = ['#ID', 'ecpp_id', 'first_name', 'last_name', race_year, 'ethnicity', 'baptismal_date',
                     'location_ecpp_baptism', 'location_ecpp_baptism', 'father_first_name', 'father_last_name',
                     'father_military_status', 'father_origin', 'mother_first_name', 'mother_last_name', 'mother_origin',
                     'sex', 'notes']
    final_output = final_output.reindex(columns=desired_order)

    logging.info(f"Finished processing matched results for {dataset_key}.")
    return final_output


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


def append_to_csv(data, filename):
    """Append data to a CSV file."""
    with open(filename, 'a') as f:
        pd.DataFrame(data).to_csv(f, header=False, index=False)


def main():
    path = '/datasets/acolinhe/data'
    output_path = '/datasets/acolinhe/data_output'
    config = get_config()
    datasets = load_and_prepare_data(path)

    all_people_collect_2 = []

    for dataset_key in datasets:
        if dataset_key == 'baptisms' or datasets[dataset_key] is None:
            continue
        results = parallel_data_processing(datasets[dataset_key], datasets['baptisms'], config, dataset_key)
        threshold = 0.87
        people_collect_2 = create_people_collect_2(results, threshold, datasets['baptisms'], datasets, dataset_key)

        if not people_collect_2.empty:
            all_people_collect_2.append(people_collect_2)

    if all_people_collect_2:
        final_people_collect_2 = pd.concat(all_people_collect_2, ignore_index=True)

        race_columns = [col for col in final_people_collect_2.columns if 'race_' in col]
        if race_columns:
            final_people_collect_2['race_aggregated'] = final_people_collect_2[race_columns].apply(
                lambda x: ' '.join(x.dropna().astype(str)), axis=1)

        final_file_path = os.path.join(output_path, 'people_collect_2.csv')
        final_people_collect_2.to_csv(final_file_path, index=False)
        logging.info(f"Final cumulative people_collect_2.csv has been saved.")


if __name__ == '__main__':
    main()
