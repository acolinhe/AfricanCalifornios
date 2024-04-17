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


def fetch_and_combine_data(match_results, data_indexed, columns):
    """Fetch data based on matches and required columns, handle missing data gracefully."""
    try:
        data = data_indexed.loc[match_results, columns]
        if data.empty:
            logging.warning("Data fetching returned an empty result.")
        return data
    except KeyError as e:
        logging.error(f"Key error during data fetching: {e}")
        return pd.DataFrame(columns=columns)


def combine_data_sets(primary_data, baptism_data):
    """Combine primary and baptism datasets into a single DataFrame, handle index reset and column renaming."""
    combined_data = pd.concat([primary_data, baptism_data], axis=0, ignore_index=True)
    return combined_data


def process_matches(matched_results, dataset_primary, dataset_baptisms, threshold, data_columns_primary,
                    data_columns_baptisms):
    """Process matching scores and extract relevant data from primary and baptisms datasets."""
    matched_data_primary = pd.DataFrame()
    matched_data_baptisms = pd.DataFrame()

    match_scores = {
        'direct': 'Direct_Total_Match_Score',
        'mother': 'Mother_Total_Match_Score',
        'father': 'Father_Total_Match_Score'
    }

    for match_type, score_column in match_scores.items():
        matches = matched_results[matched_results[score_column] >= threshold]
        if not matches.empty:
            primary_data = fetch_and_combine_data(matches['ecpp_id'], dataset_primary, data_columns_primary[match_type])
            baptism_data = fetch_and_combine_data(matches['#ID'], dataset_baptisms, data_columns_baptisms[match_type])
            matched_data_primary = pd.concat([matched_data_primary, primary_data], ignore_index=True)
            matched_data_baptisms = pd.concat([matched_data_baptisms, baptism_data], ignore_index=True)

    return matched_data_primary, matched_data_baptisms


def create_people_collect_2(matched_results, threshold, baptisms, other_datasets, dataset_key):
    logging.info(f"Starting to process matched results for {dataset_key}.")

    data_columns_primary = {
        'direct': ['Race', 'Current_Location', 'Origin Parish', 'Location Other Race']
        if '1790' in dataset_key else ['Race'],
        'mother': ['Race'],
        'father': ['Race']
    }
    data_columns_baptisms = {
        'direct': ['SpanishName', 'Surname', 'Ethnicity', 'FmtdDate', 'Mission', 'FSpanishName', 'FSurname',
                   'FMilitaryStatus', 'FOrigin', 'MSpanishName', 'MSurname', 'MOrigin', 'Sex', 'Notes'],
        'mother': ['MSpanishName', 'MSurname', 'MEthnicity', 'FmtdDate', 'Mission', 'Sex', 'Notes'],
        'father': ['FSpanishName', 'FSurname', 'FEthnicity', 'FmtdDate', 'Mission', 'Sex', 'Notes']
    }

    extracted_year = extract_year_from_filename(dataset_key)
    race_year = 'race_' + (extracted_year if extracted_year else 'unknown')
    dataset_primary = other_datasets[dataset_key].set_index('ecpp_id')
    dataset_baptisms = baptisms.set_index('#ID')

    primary_data, baptism_data = process_matches(matched_results, dataset_primary, dataset_baptisms, threshold,
                                                 data_columns_primary, data_columns_baptisms)

    final_output = pd.concat([primary_data, baptism_data], axis=1)
    final_output.rename(columns={'Race': race_year}, inplace=True)
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


def reorder_columns(dataframe):
    desired_order = [
        'last_name', 'first_name', 'race_aggregated', 'race_1790', 'race_sj1778', 'race_la1781',
        'race_la1785', 'race_la1821', 'ethnicity', 'baptismal_date', 'location_ecpp_baptism',
        'location_1790_census', 'father_last_name', 'father_first_name', 'father_military_status',
        'father_origin', 'mother_last_name', 'mother_first_name', 'mother_origin', 'sex',
        'origin_parish_1790_census', 'location_other_race', 'notes_url_1790_census'
    ]

    existing_columns = set(dataframe.columns)
    ordered_columns = [col for col in desired_order if col in existing_columns]
    return dataframe[ordered_columns]


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
        all_people_collect_2 = [df.reset_index(drop=True) for df in all_people_collect_2]
        final_people_collect_2 = pd.concat(all_people_collect_2, ignore_index=True)

        race_columns = [col for col in final_people_collect_2.columns if 'race_' in col]
        if race_columns:
            final_people_collect_2['race_aggregated'] = final_people_collect_2[race_columns].apply(
                lambda x: ' '.join(x.dropna().astype(str)), axis=1)

        reorder_columns(final_people_collect_2)

        final_file_path = os.path.join(output_path, 'people_collect_2.csv')
        final_people_collect_2.to_csv(final_file_path, index=False)
        logging.info(f"Final cumulative people_collect_2.csv has been saved.")


if __name__ == '__main__':
    main()
