import logging
import pandas as pd
from match_person_parallel_functions import load_and_prepare_data, parallel_data_processing
from data_processing import get_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def filter_matched_persons(matched_persons: pd.DataFrame, dataset_key: str, threshold: float) -> pd.DataFrame:
    match_scores = {
        'Direct_Total_Match_Score': 'direct',
        'Mother_Total_Match_Score': 'mother',
        'Father_Total_Match_Score': 'father'
    }

    columns_to_keep = ['#ID', 'ecpp_id', 'Direct_Total_Match_Score', 'Mother_Total_Match_Score',
                       'Father_Total_Match_Score']

    results = []

    for score, match_type in match_scores.items():
        filtered_persons = matched_persons[matched_persons[score] >= threshold][columns_to_keep].copy()
        filtered_persons['match_type'] = match_type
        filtered_persons['dataset_key'] = dataset_key
        results.append(filtered_persons)

    combined_df = pd.concat(results, ignore_index=True)
    logging.info(f"Filtered {len(combined_df)} persons with score >= {threshold}")

    return combined_df


def insert_matched_values(matched_persons_key: pd.DataFrame, datasets: dict) -> pd.DataFrame:
    baptisms = datasets['baptisms'].set_index('#ID')
    datasets_names = ['afro_1790_census', 'padron_1781', 'padron_1785', 'padron_1821', 'padron_267']
    for name in datasets_names:
        datasets[name] = datasets[name].set_index('ecpp_id')

    for index, row in matched_persons_key.iterrows():
        baptism_id = row['#ID']
        census_id = row['ecpp_id']

        matched_persons_key.at[index, 'baptismal_date'] = baptisms.at[baptism_id, 'Date']
        matched_persons_key.at[index, 'location_ecpp_baptism'] = baptisms.at[baptism_id, 'Mission']
        matched_persons_key.at[index, 'sex'] = baptisms.at[baptism_id, 'Sex']
        matched_persons_key.at[index, 'origin_parish_1790_census'] = baptisms.at[baptism_id, 'Place']
        matched_persons_key.at[index, 'notes_url_1790_census'] = baptisms.at[baptism_id, 'Notes']

        if row['match_type'] == 'direct':
            matched_persons_key.at[index, 'first_name'] = baptisms.at[baptism_id, 'SpanishName']
            matched_persons_key.at[index, 'last_name'] = baptisms.at[baptism_id, 'Surname']
            matched_persons_key.at[index, 'ethnicity'] = baptisms.at[baptism_id, 'Ethnicity']
            matched_persons_key.at[index, 'father_first_name'] = baptisms.at[baptism_id, 'FSpanishName']
            matched_persons_key.at[index, 'father_last_name'] = baptisms.at[baptism_id, 'FSurname']
            matched_persons_key.at[index, 'father_military_status'] = baptisms.at[baptism_id, 'FMilitaryStatus']
            matched_persons_key.at[index, 'father_origin'] = baptisms.at[baptism_id, 'FOrigin']
            matched_persons_key.at[index, 'mother_first_name'] = baptisms.at[baptism_id, 'MSpanishName']
            matched_persons_key.at[index, 'mother_last_name'] = baptisms.at[baptism_id, 'MSurname']
            matched_persons_key.at[index, 'mother_origin'] = baptisms.at[baptism_id, 'MOrigin']
            
        if row['match_type'] == 'mother':
            if row['dataset_key'] == 'afro_1790_census':
                child_columns = [f'Child{i}' for i in range(1, 15)]
                child_names = [datasets['afro_1790_census'].at[census_id, col] for col in child_columns if pd.notna(datasets['afro_1790_census'].at[census_id, col])]
                matched_persons_key.at[index, 'first_name'] = child_names[0] if child_names else ""
                matched_persons_key.at[index, 'last_name'] = child_names[1] if len(child_names) > 1 else ""
            else:
                matched_persons_key.at[index, 'first_name'] = ""
                matched_persons_key.at[index, 'last_name'] = ""
            
            matched_persons_key.at[index, 'mother_first_name'] = baptisms.at[baptism_id, 'MSpanishName']
            matched_persons_key.at[index, 'mother_last_name'] = baptisms.at[baptism_id, 'MSurname']
            matched_persons_key.at[index, 'ethnicity'] = baptisms.at[baptism_id, 'MEthnicity']

        elif row['match_type'] == 'father':
            if row['dataset_key'] == 'afro_1790_census':
                child_columns = [f'Child{i}' for i in range(1, 15)]
                child_names = [datasets['afro_1790_census'].at[census_id, col] for col in child_columns if pd.notna(datasets['afro_1790_census'].at[census_id, col])]
                matched_persons_key.at[index, 'first_name'] = child_names[0] if child_names else ""
                matched_persons_key.at[index, 'last_name'] = child_names[1] if len(child_names) > 1 else ""
            else:
                matched_persons_key.at[index, 'first_name'] = ""
                matched_persons_key.at[index, 'last_name'] = ""

            matched_persons_key.at[index, 'father_first_name'] = baptisms.at[baptism_id, 'FSpanishName']
            matched_persons_key.at[index, 'father_last_name'] = baptisms.at[baptism_id, 'FSurname']
            matched_persons_key.at[index, 'ethnicity'] = baptisms.at[baptism_id, 'FEthnicity']

        if row['dataset_key'] == 'afro_1790_census':
            matched_persons_key.at[index, 'race_1790'] = datasets['afro_1790_census'].at[census_id, 'Race']
        elif row['dataset_key'] == 'padron_1781':
            matched_persons_key.at[index, 'race_1781'] = datasets['afro_1790_census'].at[census_id, 'Race']
        elif row['dataset_key'] == 'padron_1785':
            matched_persons_key.at[index, 'race_1785'] = datasets['afro_1790_census'].at[census_id, 'Race']
        elif row['dataset_key'] == 'padron_1821':
            matched_persons_key.at[index, 'race_1821'] = datasets['afro_1790_census'].at[census_id, 'Race']
        elif row['dataset_key'] == 'padron_267':
            matched_persons_key.at[index, 'race_267'] = datasets['afro_1790_census'].at[census_id, 'Race']

    return matched_persons_key


def clean_and_create_race_aggregated(final_people_collect: pd.DataFrame) -> pd.DataFrame:
    columns_to_combine = ['race_1790', 'race_1781', 'race_1785', 'race_1821', 'race_267']

    final_people_collect['race_aggregated'] = (
        final_people_collect[columns_to_combine]
        .apply(lambda r: ', '.join(r.dropna().astype(str).str.strip().str.replace('[,\s\[\]]', ' ', regex=True)),
               axis=1)
    )

    final_people_collect['ethnicity'] = final_people_collect['ethnicity'].str.replace('[,\s\[\]]', ' ', regex=True)
    final_people_collect['origin_parish_1790_census'] = (
        final_people_collect['origin_parish_1790_census'].str.replace('[,\s\[\]]', ' ', regex=True)
    )

    final_people_collect['first_name'] = final_people_collect['first_name'].astype(str).str.replace('[', '', regex=False).str.replace(']', '', regex=False)
    final_people_collect['last_name'] = final_people_collect['last_name'].astype(str).str.replace('[', '', regex=False).str.replace(']', '', regex=False) 

    return final_people_collect


def reorder_columns(df):
    new_column_order = ['#ID', 'ecpp_id', 'Direct_Total_Match_Score', 'Mother_Total_Match_Score',
                        'Father_Total_Match_Score',
                        'match_type', 'dataset_key', 'first_name', 'last_name',
                        'father_first_name', 'father_last_name', 'father_military_status',
                        'father_origin', 'mother_first_name', 'mother_last_name',
                        'mother_origin', 'sex', 'race_1790', 'race_1781', 'race_1785',
                        'race_1821', 'race_267', 'race_aggregated', 'ethnicity',
                        'origin_parish_1790_census', 'baptismal_date', 'location_ecpp_baptism',
                        'notes_url_1790_census']

    return df[new_column_order]


def combine_parental_matches(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combines rows where individuals have been matched through both parents,
    prioritizing data from direct matches.
    """

    def combine_rows(group):
        direct_match = group[group['match_type'] == 'Direct']
        if not direct_match.empty:
            combined_row = direct_match.iloc[0].copy()
        else:
            combined_row = group.iloc[0].copy()

        if len(group) > 1:
            for col in ['Direct_Total_Match_Score', 'Mother_Total_Match_Score', 'Father_Total_Match_Score']:
                combined_row[col] = group[col].max()
            
            # Combine data from other columns, prioritizing direct match if available
            for col in group.columns:
                if col not in ['Direct_Total_Match_Score', 'Mother_Total_Match_Score', 'Father_Total_Match_Score', 'match_type', '#ID', 'ecpp_id']:
                    direct_value = direct_match[col].values[0] if not direct_match.empty else None
                    mother_value = group[group['match_type'] == 'mother'][col].values[0] if not group[group['match_type'] == 'mother'].empty else None
                    father_value = group[group['match_type'] == 'father'][col].values[0] if not group[group['match_type'] == 'father'].empty else None

                    combined_row[col] = direct_value or mother_value or father_value  # Prioritize direct, then mother, then father

            combined_row['match_type'] = 'mother_father'  # Update match type
        return combined_row

    df = df.groupby(['#ID', 'ecpp_id']).apply(combine_rows).reset_index(drop=True)
    return df


def main():
    path = '/home/acolinhe/AfricanCalifornios/matchVersion1/data'
    output_path = '/home/acolinhe/AfricanCalifornios/matchVersion1/data_output'
    config = get_config()
    datasets = load_and_prepare_data(path)
    matched_persons_key = []

    for dataset_key in datasets:
        if dataset_key == 'baptisms' or datasets[dataset_key] is None:
            continue
        matched_persons = parallel_data_processing(datasets[dataset_key], datasets['baptisms'], config, dataset_key)
        matched_persons_key.append(filter_matched_persons(matched_persons, dataset_key, .45))
        logging.info(f"Completed filtering {dataset_key}")

    combined_matched_persons_key = pd.concat(matched_persons_key, ignore_index=True)
    
    combined_matched_persons_key = combine_parental_matches(combined_matched_persons_key)  

    people_collect_2 = insert_matched_values(combined_matched_persons_key, datasets)
    cleaned_people_collect_2 = clean_and_create_race_aggregated(people_collect_2)
    final_people_collect_2 = reorder_columns(cleaned_people_collect_2)

    final_people_collect_2.to_csv(output_path + '/people_collect_2.csv', index=False)
    logging.info(f"people_collect_2.csv columns {final_people_collect_2.columns}")
    logging.info(f"Completed matches and saved to {output_path + '/people_collect_2.csv'}")


if __name__ == '__main__':
    main()