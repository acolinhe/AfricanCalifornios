import os
import logging
import pandas as pd
import time
from multiprocessing import Pool, cpu_count
from data_processing import add_identifiers, clean_names_padrones
from data_loading import load_data, create_afro_df
from person_matcher import PersonMatcher

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def configure_and_match(census, baptisms, config):
    """Configure the datasets and perform matching."""
    matcher = PersonMatcher(census=census, baptisms=baptisms, config=config)
    matcher.match()
    return matcher.create_matched_records()


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
    """Load data and prepare it by adding identifiers and cleaning names."""
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
            add_identifiers(dataset, 'ecpp_id' if 'padron' in name or 'census' in name else '#ID')
            if 'padron' in name:
                clean_names_padrones(dataset)
    return datasets


def save_to_pickle(dataframe, path, filename):
    """Save the dataframe to a pickle file."""
    full_path = os.path.join(path, filename)
    dataframe.to_pickle(full_path)
    logging.info(f"Dataframe saved to {full_path}")


def main():
    path = '/datasets/acolinhe/data'
    output_path = '/datasets/acolinhe/data_output'
    config = {
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

    datasets = load_and_prepare_data(path)
    num_cores = min(32, cpu_count())

    for dataset_key in datasets:
        if dataset_key == 'baptisms' or datasets[dataset_key] is None:
            continue
        dataset_start_time = time.time()
        pool = Pool(processes=num_cores)
        try:
            chunks = chunk_dataframe(datasets[dataset_key], num_cores)
            current_config = config['padrones_config'] if 'padron' in dataset_key else config['census_config']
            tasks = [(chunk, datasets['baptisms'], current_config, index) for chunk, index in chunks]

            results = pool.map(parallel_configure_and_match, tasks)
        finally:
            pool.close()
            pool.join()

        matched_results = pd.concat(results)
        save_to_pickle(matched_results, output_path, f'{dataset_key}_matched.pkl')
        logging.info(f"Total matched records for {dataset_key}: {len(matched_results)}")
        logging.info(f"{dataset_key} processing time: {time.time() - dataset_start_time} seconds")


if __name__ == '__main__':
    main()
