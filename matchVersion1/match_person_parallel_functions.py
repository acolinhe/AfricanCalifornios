import logging
import numpy as np
import pandas as pd
import time
from data_processing import add_identifiers, clean_names_padrones, get_datasets
from multiprocessing import Pool, cpu_count
from person_matcher import PersonMatcher


def load_and_prepare_data(path: str) -> dict:
    """Load data and prepare it by adding identifiers and cleaning names, ensuring 'Age' column exists."""
    datasets = get_datasets(path)
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


def chunk_dataframe(df: pd.DataFrame, num_chunks: int) -> list:
    """Chunk the dataframe into smaller parts, returning chunks along with their indices."""
    chunk_size = len(df) // num_chunks + (len(df) % num_chunks > 0)
    return [(df.iloc[i:i + chunk_size], i) for i in range(0, len(df), chunk_size)]


def configure_and_match(census: pd.DataFrame, baptisms: pd.DataFrame, config: dict) -> PersonMatcher:
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