from person_matching_functions import *
import pandas as pd
from itertools import product
import numpy as np


# This will probably take a while to run
# Trying to make generic class for baptisms, marriages, and deaths
# Might have to make a separate class for each
class PersonMatcher:
    def __init__(self, ecpp: pd.DataFrame, baptisms: pd.DataFrame):
        self.ecpp = ecpp
        self.records = baptisms
        self.ecpp_id_col = 'ecpp_id'
        self.records_id_col = '#ID'
        self.matched_records = pd.DataFrame(columns=[self.ecpp_id_col, self.records_id_col])

    def create_matched_records(self):
        ecpp_ids = self.ecpp[self.ecpp_id_col]
        records_ids = self.records[self.records_id_col].astype(int)
        
        all_pairs = product(ecpp_ids, records_ids)
        
        matched_records_list = [{'ecpp_id': ecpp_id, '#ID': record_id} for ecpp_id, record_id in all_pairs]
        self.matched_records = pd.DataFrame(matched_records_list, columns=[self.ecpp_id_col, self.records_id_col])
    
    def direct_name_matcher(self):
        ecpp_first_name_values = self.ecpp.set_index(self.ecpp_id_col)['First'].to_dict()
        records_first_name_values = self.records.set_index(self.records_id_col)['SpanishName'].to_dict()
        ecpp_last_name_values = self.ecpp.set_index(self.ecpp_id_col)['Last'].to_dict()
        records_last_name_values = self.records.set_index(self.records_id_col)['Surname'].to_dict()

        self.matched_records['Census First Name'] = self.matched_records[self.ecpp_id_col].map(ecpp_first_name_values)
        self.matched_records['Baptisms First Name'] = self.matched_records[self.records_id_col].map(records_first_name_values)
        self.matched_records['First Name Match Classifier'] = 0

        self.matched_records['Census Last Name'] = self.matched_records[self.ecpp_id_col].map(ecpp_last_name_values)
        self.matched_records['Baptisms Last Name'] = self.matched_records[self.records_id_col].map(records_last_name_values)
        self.matched_records['Last Name Match Classifier'] = 0
    
    def direct_match_names(self):
        self.matched_records['First Name Match Classifier'] = \
        self.matched_records.apply(lambda row: match_names_score(row['Census First Name'], row['Baptisms First Name']), axis=1)
        self.matched_records['Last Name Match Classifier'] = \
        self.matched_records.apply(lambda row: match_names_score(row['Census Last Name'], row['Baptisms Last Name']), axis=1)

    # Now for the parent matching
    # read in the pickle file to save on matching time
    def match_parents(self):
        records_mother_first_name_values = self.records.set_index(self.records_id_col)['MSpanishName'].to_dict()
        records_mother_last_name_values = self.records.set_index(self.records_id_col)['MSurname'].to_dict()
        records_father_first_name_values = self.records.set_index(self.records_id_col)['FSpanishName'].to_dict()
        records_father_last_name_values = self.records.set_index(self.records_id_col)['FSurname'].to_dict()


        self.matched_records['Mother Baptisms First Name'] = self.matched_records[self.records_id_col].map(records_mother_first_name_values)
        self.matched_records['Mother First Name Match Classifier'] = 0
        self.matched_records['Mother Baptisms Last Name'] = self.matched_records[self.records_id_col].map(records_mother_last_name_values)
        self.matched_records['Mother Last Name Match Classifier'] = 0

        self.matched_records['Father Baptisms First Name'] = self.matched_records[self.records_id_col].map(records_father_first_name_values)
        self.matched_records['Father First Name Match Classifier'] = 0
        self.matched_records['Father Baptisms Last Name'] = self.matched_records[self.records_id_col].map(records_father_last_name_values)
        self.matched_records['Father Last Name Match Classifier'] = 0
    
    def match_parents_names(self):
        self.matched_records['Mother First Name Match Classifier'] = \
        self.matched_records.apply(lambda row: match_names_score(row['Census First Name'], row['Mother Baptisms First Name']), axis=1)
        self.matched_records['Mother Last Name Match Classifier'] = \
        self.matched_records.apply(lambda row: match_names_score(row['Census Last Name'], row['Mother Baptisms Last Name']), axis=1)

        self.matched_records['Father First Name Match Classifier'] = \
        self.matched_records.apply(lambda row: match_names_score(row['Census First Name'], row['Father Baptisms First Name']), axis=1)
        self.matched_records['Father Last Name Match Classifier'] = \
        self.matched_records.apply(lambda row: match_names_score(row['Census Last Name'], row['Father Baptisms Last Name']), axis=1)
    
    def match_other_features(self):
        ecpp_gender_values = self.ecpp.set_index(self.ecpp_id_col)['Gender'].to_dict()
        records_gender_values = self.records.set_index(self.records_id_col)['Sex'].to_dict()

        ecpp_age_values = self.ecpp.set_index(self.ecpp_id_col)['Age'].to_dict()
        records_age_values = self.records.set_index(self.records_id_col)['Age'].to_dict()

        self.matched_records['Census Gender'] = self.matched_records[self.ecpp_id_col].map(ecpp_gender_values).str.lower()
        self.matched_records['Baptisms Gender'] = self.matched_records[self.records_id_col].map(records_gender_values).str.lower()

        self.matched_records['Gender Match Classifier'] = np.where(
        self.matched_records['Census Gender'] == self.matched_records['Baptisms Gender'], 1, 0)

        self.matched_records['Census Age'] = self.matched_records[self.ecpp_id_col].map(ecpp_age_values)
        self.matched_records['Baptisms Age'] = self.matched_records[self.records_id_col].map(records_age_values)
        self.matched_records['Age Match Range'] = abs(self.matched_records['Census Age'] - self.matched_records['Baptisms Age']) 
        

    def save_to_pickle(self, filename):
        self.matched_records.to_pickle(filename)

    def read_pickle(self, filename):
        self.matched_records = pd.read_pickle(filename)

