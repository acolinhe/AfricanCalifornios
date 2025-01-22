mappings = {
    "1790_census": {
        "name": ["first", "last"],
        "race": "race",
        "gender": "gender",
        "father": ["father_first", "father_last"],
        "mother": ["mother_first", "mother_last"],
        "spouse": ["spouse_first", "spouse_last"]
    },
    "baptisms": {
        "name": ["mspanishname", "msurname"],  # Mother's SpanishName and Surname
        "race": "origin",  # Origin column indicates race
        "gender": "sex",  # Gender available as Sex
        "father": ["fspanishname", "fsurname"],  # Father's SpanishName and Surname
        "mother": ["mspanishname", "msurname"],  # Mother's SpanishName and Surname
        "spouse": [None, None]  # Spouse information is not available
    },
    "padron_1767": {
        "name": ["ego_first name", "ego_paternal last name"],  # Combine first and last name
        "race": "race",
        "gender": "sex",
        "father": ["father_first name ", "father_paternal last name"],  # Combine father's first and last names
        "mother": ["mother_first name", "mother_paternal last name"],  # Combine mother's first and last names
        "spouse": ["husband_first name ", "huband_paternal last name"]  # Combine husband's first and last names
    },
    "padron_1781": {
        "name": ["ego_first name", "ego_paternal last name"],
        "race": "race",
        "gender": "sex",
        "father": ["father_first name", "father_paternal last name"],
        "mother": ["mother_first name", "mother_paternal last name"],
        "spouse": ["husband_first name", "huband_paternal last name"]
    },
    "padron_1785": {
        "name": ["ego_first name", "ego_paternal last name"],
        "race": "race",
        "gender": "sex",
        "father": ["father_first name", "father_paternal last name"],
        "mother": ["mother_first name", "mother_paternal last name"],
        "spouse": ["husband_first name", "huband_paternal last name"],
        "marital_status": "marital status",  # Additional marital status column
        "profession": "profession"  # Additional profession column
    },
    "padron_1821": {
        "name": ["ego_first name", "ego_paternal last name"],
        "race": "(color) race",  # Updated to match '(Color) Race'
        "gender": "sex",
        "father": ["father_first name", "father_paternal last name"],
        "mother": ["mother_first name", "mother_paternal last name"],
        "spouse": ["husband_first name", "huband_paternal last name"],
        "birth_year": "birth year (est.)"  # Additional birth year column
    }
}
