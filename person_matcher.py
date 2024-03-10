from person_matching_functions import *
import pandas as pd
from itertools import product
import numpy as np


class PersonMatcher:
    def __init__(self, census, baptisms, config):
        self.ecpp = census
        self.records = baptisms
        self.config = config
        self.matched_records = pd.DataFrame()
    
    def calculate_total_match_score(self):
        weights = {
            'First_Name_Match_Score': 0.20,  # Updated weight for first name
            'Last_Name_Match_Score': 0.35,   # Updated weight for last name
            'Gender_Match_Score': 0.30,
            'Age_Match_Score': 0.15
        }

        weight_sum = sum(weights.values())
        normalized_weights = {k: v / weight_sum for k, v in weights.items()}

        self.matched_records['Total_Match_Score'] = sum(
            self.matched_records[score] * normalized_weights[score]
            for score in weights.keys()
        )


    def match(self):
        self.create_matched_records()
        self.direct_match_names()
        self.match_parents_names()
        self.match_other_features()
        self.calculate_total_match_score()
        return self.matched_records

    def create_matched_records(self):
        ecpp_ids = self.ecpp.reset_index()[self.config['ecpp_id_col']]
        records_ids = self.records.reset_index()[self.config['records_id_col']]

        self.matched_records = pd.DataFrame(product(ecpp_ids, records_ids), columns=[self.config['ecpp_id_col'], self.config['records_id_col']])

        for key, value in self.config['census'].items():
            self.matched_records[f'Census_{value}'] = self.matched_records[self.config['ecpp_id_col']].map(self.ecpp.set_index(self.config['ecpp_id_col'])[value])

        for key, value in self.config['baptisms'].items():
            self.matched_records[f'Baptisms_{value}'] = self.matched_records[self.config['records_id_col']].map(self.records.set_index(self.config['records_id_col'])[value])
        
        return self.matched_records


    def direct_match_names(self):
        # Calculate and assign first name match score
        self.matched_records['First_Name_Match_Score'] = self.matched_records.apply(
            lambda row: self.match_name_score(
                row[f'Census_{self.config["census"]["First Name"]}'],
                row[f'Baptisms_{self.config["baptisms"]["First Name"]}']
            ) * 0.20, axis=1
        )

        # Calculate and assign last name match score
        self.matched_records['Last_Name_Match_Score'] = self.matched_records.apply(
            lambda row: self.match_name_score(
                row[f'Census_{self.config["census"]["Last Name"]}'],
                row[f'Baptisms_{self.config["baptisms"]["Last Name"]}']
            ) * 0.35, axis=1
        )


    def match_parents_names(self):
        self.matched_records['Parent_Name_Match_Score'] = self.matched_records.apply(
            lambda row: max(
                self.match_name_score(
                    row[f'Census_{self.config["census"]["First Name"]}'],
                    row[f'Baptisms_{self.config["baptisms"]["Mother First Name"]}']
                ) * self.match_name_score(
                    row[f'Census_{self.config["census"]["Last Name"]}'],
                    row[f'Baptisms_{self.config["baptisms"]["Mother Last Name"]}']
                ),
                self.match_name_score(
                    row[f'Census_{self.config["census"]["First Name"]}'],
                    row[f'Baptisms_{self.config["baptisms"]["Father First Name"]}']
                ) * self.match_name_score(
                    row[f'Census_{self.config["census"]["Last Name"]}'],
                    row[f'Baptisms_{self.config["baptisms"]["Father Last Name"]}']
                )
            ), axis=1
        )

    
    def match_other_features(self):
        self.matched_records['Gender_Match_Score'] = self.matched_records.apply(
            lambda row: self.match_gender_score(
                row[f'Census_{self.config["census"]["Gender"]}'],
                row[f'Baptisms_{self.config["baptisms"]["Gender"]}']
            ), axis=1
        )
        self.matched_records['Age_Match_Score'] = self.matched_records.apply(
            lambda row: self.match_age_score(
                row[f'Census_{self.config["census"]["Age"]}'],
                row[f'Baptisms_{self.config["baptisms"]["Age"]}']
            ), axis=1
        )


    def match_name_score(self, name_census, name_baptism):
        return 1 - np.exp(-match_names_score(name_census, name_baptism))

    def match_gender_score(self, gender_census, gender_baptism):
        if pd.isna(gender_census) or pd.isna(gender_baptism):
            return 0  # Return a score of 0 if either input is missing

        try:
            gender_census = str(gender_census).lower()
            gender_baptism = str(gender_baptism).lower()
        except AttributeError:
            return 0  # Return a score of 0 if either cannot be converted to string and lowered

        return 1 if gender_census == gender_baptism else 0

    def match_age_score(self, age_census, age_baptism):
        if pd.isna(age_census) or pd.isna(age_baptism):
            return 0.0

        max_age_diff = 5
        try:
            age_diff = abs(float(age_census) - float(age_baptism))
            return max(0, (max_age_diff - age_diff) / max_age_diff)
        except ValueError:
            return 0.0 
    
    def save_matched_records(self, filename):
        self.matched_records.to_pickle(filename)

