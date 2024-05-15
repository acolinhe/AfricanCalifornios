import logging
from typing import Optional
import pandas as pd
from family_unit import FamilyUnit, PersonUnit

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def read_people_collect() -> Optional[pd.DataFrame]:
    """ Load data from a CSV file and handle potential errors. """
    people_collect_path = '/datasets/acolinhe/data_output/people_collect_2.csv'
    try:
        return pd.read_csv(people_collect_path)
    except FileNotFoundError:
        logging.error(f"File not found: {people_collect_path}")
        return None
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return None


def main():
    # put in main for now and abstract away with functions
    pc2 = read_people_collect()
    print(pc2)
    family = FamilyUnit(0)


if __name__ == '__main__':
    main()
