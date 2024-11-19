import pandas as pd
import re

def ensembleVote(row):
    """
    Combines the 'Persons (entities)' predictions from different NER models
    by taking the union of all names.
    """
    personsFlair = set(row['Persons (entities)_flair'].split(', ')) if isinstance(row['Persons (entities)_flair'], str) else set()
    personsHuggingface = set(row['Persons (entities)_huggingface'].split(', ')) if isinstance(row['Persons (entities)_huggingface'], str) else set()
    personsSpacy = set(row['Persons (entities)'].split(', ')) if isinstance(row['Persons (entities)'], str) else set()
    personsStanza = set(row['Persons (entities)_stanza'].split(', ')) if isinstance(row['Persons (entities)_stanza'], str) else set()

    # Combine all entities from the different models
    allPersons = personsFlair | personsHuggingface | personsSpacy | personsStanza

    # Remove unwanted characters from each person's name
    cleanedPersons = []
    for person in allPersons:
        cleanedPerson = re.sub(r"[\[\]'\"']", "", person).strip()  # Strip [, ], ', " and whitespace
        cleanedPersons.append(cleanedPerson)

    # Return a comma-separated string of all names
    return ', '.join(cleanedPersons)


def createEnsemble(flairCsv, huggingfaceCsv, spacyCsv, stanzaCsv, outputCsv):
    """
    Creates an ensemble NER model by combining the results from different NER models
    stored in CSV files.
    """
    dfFlair = pd.read_csv(flairCsv)
    dfHuggingface = pd.read_csv(huggingfaceCsv)
    dfSpacy = pd.read_csv(spacyCsv)
    dfStanza = pd.read_csv(stanzaCsv)

    # Merge the DataFrames (using 'Grant Name' and 'Grant Number' as keys)
    mergedDf = dfFlair.merge(dfHuggingface, on=['Grant Name', 'Grant Number'], suffixes=('_flair', '_huggingface'))

    # Merge df_spacy without renaming its columns
    mergedDf = mergedDf.merge(dfSpacy, on=['Grant Name', 'Grant Number'])

    # Merge df_stanza without renaming its columns, but rename the column in df_stanza
    mergedDf = mergedDf.merge(dfStanza.rename(columns={'Persons (entities)': 'Persons (entities)_stanza'}),
                              on=['Grant Name', 'Grant Number'])

    mergedDf['Ensemble_Persons'] = mergedDf.apply(ensembleVote, axis=1)
    mergedDf.to_csv(outputCsv, index=False)


if __name__ == "__main__":
    outputDir = "../data/"
    flairCsv = "landGrantFlair.csv"
    huggingfaceCsv = "landGrantHuggingFace.csv"
    spacyCsv = "landGrantSpacy.csv"
    stanzaCsv = "landGrantStanza.csv"
    outputCsv = "landGrantEnsemble.csv"

    createEnsemble(outputDir + flairCsv, outputDir + huggingfaceCsv, outputDir + spacyCsv, outputDir + stanzaCsv, outputDir + outputCsv)