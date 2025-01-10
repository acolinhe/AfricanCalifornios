from person_matching_functions import *
import pandas as pd
from itertools import product
import logging

class PersonMatcher:
    def __init__(self, census, baptisms, config):
        self.ecpp = census
        self.records = baptisms
        self.config = config
        self.matched_records = pd.DataFrame()

    def match(self):
        self.create_matched_records()
        self.direct_match_names()
        self.match_parents_names()
        self.match_other_features()
        self.calculate_total_match_score('Direct_Total_Match_Score')
        self.calculate_total_match_score('Mother_Total_Match_Score')
        self.calculate_total_match_score('Father_Total_Match_Score')
        self.list_matched_criteria()
        return

    def calculate_total_match_score(self, score_column_name):
        self.matched_records[score_column_name] = 0
        for score_column in ["First_Name_Match_Score", "Last_Name_Match_Score", "Gender_Match_Score", "Age_Match_Score"]:
            if score_column in self.matched_records.columns:
                self.matched_records[score_column_name] += self.matched_records[score_column]

    def create_matched_records(self):
        ecpp_ids = self.ecpp.reset_index()[self.config['ecpp_id_col']]
        records_ids = self.records.reset_index()[self.config['records_id_col']]

        self.matched_records = pd.DataFrame(product(ecpp_ids, records_ids),
                                            columns=[self.config['ecpp_id_col'], self.config['records_id_col']])

        for key, value in self.config['census'].items():
            self.matched_records[f'Census_{value}'] = self.matched_records[self.config['ecpp_id_col']].map(
                self.ecpp.set_index(self.config['ecpp_id_col'])[value])

        for key, value in self.config['baptisms'].items():
            self.matched_records[f'Baptisms_{value}'] = self.matched_records[self.config['records_id_col']].map(
                self.records.set_index(self.config['records_id_col'])[value])

        return self.matched_records

    def direct_match_names(self):
        self.matched_records['First_Name_Match_Score'] = self.matched_records.apply(
            lambda row: self.match_name_score(
                row[f'Census_{self.config["census"]["First Name"]}'],
                row[f'Baptisms_{self.config["baptisms"]["First Name"]}']
            ), axis=1
        )

        self.matched_records['Last_Name_Match_Score'] = self.matched_records.apply(
            lambda row: self.match_name_score(
                row[f'Census_{self.config["census"]["Last Name"]}'],
                row[f'Baptisms_{self.config["baptisms"]["Last Name"]}']
            ), axis=1
        )

    def match_parents_names(self):
        self.matched_records['Mother_First_Name_Match_Score'] = self.matched_records.apply(
            lambda row: self.match_name_score(
                row[f'Census_{self.config["census"]["First Name"]}'],
                row[f'Baptisms_{self.config["baptisms"]["Mother First Name"]}']
            ), axis=1
        )
        self.matched_records['Mother_Last_Name_Match_Score'] = self.matched_records.apply(
            lambda row: self.match_name_score(
                row[f'Census_{self.config["census"]["Last Name"]}'],
                row[f'Baptisms_{self.config["baptisms"]["Mother Last Name"]}']
            ), axis=1
        )
        self.matched_records['Father_First_Name_Match_Score'] = self.matched_records.apply(
            lambda row: self.match_name_score(
                row[f'Census_{self.config["census"]["First Name"]}'],
                row[f'Baptisms_{self.config["baptisms"]["Father First Name"]}']
            ), axis=1
        )
        self.matched_records['Father_Last_Name_Match_Score'] = self.matched_records.apply(
            lambda row: self.match_name_score(
                row[f'Census_{self.config["census"]["Last Name"]}'],
                row[f'Baptisms_{self.config["baptisms"]["Father Last Name"]}']
            ), axis=1
        )

    def match_other_features(self):
        if 'Age' in self.config['census'] and f'Census_{self.config["census"]["Age"]}' in self.ecpp.columns:
            self.matched_records['Age_Match_Score'] = self.matched_records.apply(
                lambda row: self.match_age_score(
                    row[f'Census_{self.config["census"]["Age"]}'],
                    row[f'Baptisms_{self.config["baptisms"]["Age"]}']
                ), axis=1
            )
        else:
            self.matched_records['Age_Match_Score'] = 0  # Default to 0 if age data is missing

        self.matched_records['Gender_Match_Score'] = self.matched_records.apply(
            lambda row: self.match_gender_score(
                row[f'Census_{self.config["census"]["Gender"]}'],
                row[f'Baptisms_{self.config["baptisms"]["Gender"]}']
            ), axis=1
        )

    def match_name_score(self, name_census, name_baptism):
        if pd.isna(name_census) or pd.isna(name_baptism):
            return 0
        distance = match_names_score(name_census, name_baptism)
        return 1 if distance <= 2 else 0

    def match_gender_score(self, gender_census, gender_baptism):
        if pd.isna(gender_census) or pd.isna(gender_baptism):
            return 0

        try:
            gender_census = str(gender_census).lower()
            gender_baptism = str(gender_baptism).lower()
        except AttributeError:
            return 0

        return 1 if gender_census == gender_baptism else 0

    def match_age_score(self, age_census, age_baptism):
        if pd.isna(age_census) or pd.isna(age_baptism):
            return 0
        try:
            age_diff = abs(float(age_census) - float(age_baptism))
            return 1 if age_diff <= 2 else 0  # Tolerance of +/- 2 years
        except ValueError:
            return 0

    def get_matched_criteria(self, row):  # Corrected indentation
        criteria = []
        if row['First_Name_Match_Score'] == 1:
            criteria.append('First Name')
        if row['Last_Name_Match_Score'] == 1:
            criteria.append('Last Name')
        if row['Age_Match_Score'] == 1:
            criteria.append('Age')
        if row['Gender_Match_Score'] == 1:
            criteria.append('Gender')
        return ', '.join(criteria)

    def list_matched_criteria(self):
        self.matched_records['Matched_Criteria'] = self.matched_records.apply(self.get_matched_criteria, axis=1)

    def save_matched_records(self, filename):
        self.matched_records.to_pickle(filename)