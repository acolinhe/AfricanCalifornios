import os
import logging
import pandas as pd
import time
import numpy as np
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


def create_people_collect_2(matched_results, threshold, baptisms, other_datasets, dataset_key):
    logging.info(f"Starting to process matched results for {dataset_key}.")

    if 'padron' in dataset_key or 'census' in dataset_key:
        index_key = 'ecpp_id'
        dataset = other_datasets[dataset_key]
        data_columns = {
            'direct': ['Race'],
            'mother': ['Race'],
            'father': ['Race']
        }
    else:
        index_key = '#ID'
        dataset = baptisms
        data_columns = {
            'direct': ['SpanishName', 'Surname'],
            'mother': ['MSpanishName', 'MSurname'],
            'father': ['FSpanishName', 'FSurname']
        }

    dataset_indexed = dataset.set_index(index_key)
    logging.info(f"{dataset_key} data indexed by {index_key}.")

    direct_matches = matched_results[matched_results['Direct_Total_Match_Score'] >= threshold]
    mother_matches = matched_results[matched_results['Mother_Total_Match_Score'] >= threshold]
    father_matches = matched_results[matched_results['Father_Total_Match_Score'] >= threshold]

    direct_data = dataset_indexed.loc[direct_matches[index_key], data_columns['direct']]
    mother_data = dataset_indexed.loc[mother_matches[index_key], data_columns['mother']]
    father_data = dataset_indexed.loc[father_matches[index_key], data_columns['father']]

    combined = pd.concat([direct_data, mother_data, father_data], axis=0)
    combined[index_key] = combined.index

    combined = combined.reset_index(drop=True)

    combined.rename(columns={'Race': 'race_aggregated'}, inplace=True)

    if 'padron' in dataset_key or 'census' in dataset_key:
        required_columns = ['race_aggregated']
    else:
        combined.rename(columns={'SpanishName': 'first_name', 'Surname': 'last_name', 'MSpanishName': 'first_name',
                                 'MSurname': 'last_name', 'FSpanishName': 'first_name', 'FSurname': 'last_name'},
                        inplace=True)
        required_columns = ['first_name', 'last_name']

    final_output = combined[[index_key] + required_columns]

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
        final_people_collect_2.to_csv(os.path.join(output_path, 'people_collect_2.csv'), index=False)
        logging.info(f"Final cumulative people_collect_2.csv has been saved.")


if __name__ == '__main__':
    main()