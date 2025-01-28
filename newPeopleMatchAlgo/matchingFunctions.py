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


def modified_levenshtein_distance(name1: str, name2: str, cost_dict: dict=custom_costs):
    """
    Calculates a modified Levenshtein distance between two names, allowing for custom substitution costs,
    and returns early if the distance exceeds a dynamic maximum distance.

    Args:
        name1 (str): The first name.
        name2 (str): The second name.
        cost_dict (dict): A dictionary of substitution costs {(char1, char2): cost}.

    Returns:
        int: The modified Levenshtein distance between the two names with custom costs into account and returns a score.
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


        previous_row = current_row

    return previous_row[-1]


# def convert_age(x):
#     """
#     Converts age into a float, handling potential type inconsistencies.

#     Args:
#         x: The input age value.

#     Returns:
#         float: The converted age, or 0 if conversion fails.
#         None: If the input is already None.
#     """

#     if pd.isna(x):
#         return x

#     elif isinstance(x, str):
#         try:
#             return float(x)
#         except ValueError:
#             return 0
#     else:
#         return x