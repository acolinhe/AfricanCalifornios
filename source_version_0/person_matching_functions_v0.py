import pandas as pd
import unicodedata
import numpy as np

custom_costs = {
    ('b', 'v'): 0, ('v', 'b'): 0, 
    ('c', 's'): 0, ('s', 'c'): 0, ('c', 'z'): 0, ('z', 'c'): 0, ('s', 'z'): 0, ('z', 's'): 0,
    ('i', 'y'): 0, ('y', 'i'): 0, 
    ('g', 'j'): 0, ('j', 'g'): 0, 
    ('c', 'q'): 0, ('q', 'c'): 0, 
    ('u', 'v'): 0, ('v', 'u'): 0 
}

def normalize_spanish_names(name: str):
    """
    Normalizes Spanish names by removing diacritics and accents, and replacing 
    'ñ' with 'n'.

    Args:
        name (str): The original Spanish name.

    Returns:
        str: The normalized Spanish name in lowercase.
    """

    if pd.isnull(name) or not isinstance(name, str):
        return name

    unicode_decomp = unicodedata.normalize('NFKD', name) 

    normalized_name = ''.join(
        char for char in unicode_decomp
        if not unicodedata.combining(char)
    ) 

    normalized_name = normalized_name.replace('ñ', 'n').replace('Ñ', 'N')

    return normalized_name.lower()


def modified_levenshtein_distance(name1: str, name2: str, cost_dict: dict, dynamic_max_distance: int):
    """
    Calculates a modified Levenshtein distance between two names, allowing for custom substitution costs,
    and returns early if the distance exceeds a dynamic maximum distance.

    Args:
        name1 (str): The first name.
        name2 (str): The second name.
        cost_dict (dict): A dictionary of substitution costs {(char1, char2): cost}.
        dynamic_max_distance (int): The dynamic maximum distance to consider. If the calculated distance
                                    exceeds this value, the function returns early.

    Returns:
        int: The modified Levenshtein distance between the two names, or an indication that the distance
             exceeds the dynamic maximum distance.
    """

    if len(name1) < len(name2):
        name1, name2 = name2, name1

    if not name2:
        return len(name1)

    previous_row = np.arange(len(name2) + 1)
    for i, char1 in enumerate(name1):
        current_row = np.zeros(len(name2) + 1, dtype=int)
        current_row[0] = i + 1
        for j, char2 in enumerate(name2):
            insertion_cost = previous_row[j + 1] + 1
            deletion_cost = current_row[j] + 1
            if char1 == char2:
                substitution_cost = previous_row[j]
            else:
                substitution_cost = previous_row[j] + cost_dict.get((char1, char2), 1)
            current_row[j + 1] = min(insertion_cost, deletion_cost, substitution_cost)
        
        # Check if the minimum distance so far exceeds the dynamic_max_distance
        if min(current_row) > dynamic_max_distance:
            return dynamic_max_distance + 1  # Indicates exceeding the threshold
        
        previous_row = current_row

    return previous_row[-1]


def convert_age(x):
    """
    Converts age into a float, handling potential type inconsistencies.

    Args:
        x: The input age value.

    Returns:
        float: The converted age, or 0 if conversion fails.
        None: If the input is already None.
    """

    if pd.isna(x):
        return x

    elif isinstance(x, str):
        try:
            return float(x)
        except ValueError:
            return 0
    else:
        return x


def calculate_match_probability(levenshtein_distance, max_distance):
    """
    Calculates a match probability score based on the Levenshtein distance.

    Args:
        levenshtein_distance (int): The Levenshtein distance between two names.
        max_distance (int): The maximum Levenshtein distance considered for matching.

    Returns:
        float: A probability score between 0 and 1, with 1 being a perfect match
               and 0 being no match.
    """
    
    if levenshtein_distance is None:
        return 0
    
    if levenshtein_distance > max_distance:
        return 0
    
    return (max_distance - levenshtein_distance) / max_distance


def match_names_score(name1, name2, max_distance=5):
    """
    Returns a score for levenshteing distance. 

    Args:
        name1 (str): Given name for matching.
        name2 (str): Given name for matching.
        max_distance (int): Minimum levenshtein distance for matching.
    
    Returns:
        float: Probability score between 0 and 1 for levenshtein distance matching.

    """
    name1 = str(name1) if not pd.isna(name1) else ""
    name2 = str(name2) if not pd.isna(name2) else ""

    name1 = normalize_spanish_names(name1)
    name2 = normalize_spanish_names(name2)

    dynamic_max = dynamic_max_distance(name1, name2)

    levenshtein_distance = modified_levenshtein_distance(name1, name2, custom_costs, dynamic_max)

    return calculate_match_probability(levenshtein_distance, dynamic_max)


def dynamic_max_distance(name1, name2):
    base_max_distance = 5
    length_threshold = 8
    extra_length_factor = 0.2
    
    max_len = max(len(name1), len(name2))
    
    if max_len <= length_threshold:
        return base_max_distance
    else:
        extra_length = max_len - length_threshold
        return base_max_distance + (extra_length * extra_length_factor)



def filter_records_by_score(matched_records, score_name,  threshold):
    return matched_records[matched_records[score_name] > threshold]

