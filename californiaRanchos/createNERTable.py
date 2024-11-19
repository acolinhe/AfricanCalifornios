import pandas as pd
import os
import matplotlib.pyplot as plt
import re


def loadCorpus(filePath):
    """Loads the text corpus and returns the total word count."""
    with open(filePath, 'r') as file:
        text = file.read()
    wordCount = len(text.split())
    return wordCount


def calculateTotalWords(df, column):
    """Calculates the total word count from names in a specified column across all rows in the DataFrame."""
    if column not in df.columns:
        print(f"Warning: Column '{column}' not found in DataFrame. Skipping this file.")
        return 0
    totalWords = df[column].apply(lambda x: len(x.split(',')) if pd.notna(x) else 0).sum()
    return totalWords


def calculateCorrectMatches(df):
    """Calculates the total correct matches from the 'Match' column if it exists."""
    if 'Match' not in df.columns:
        return 0
    return df['Match'].sum()


def calculateConfusionMatrix(df):
    """Calculate True Positives, False Positives, False Negatives, and True Negatives."""
    confusionMatrixStats = []

    for modelName, modelDf in df.groupby('Model'):
        TP = modelDf['True Positives'].sum()
        FP = modelDf['False Positives'].sum()
        FN = modelDf['False Negatives'].sum()

        totalEntities = modelDf['Total Names in Generated'].sum() + modelDf['Total Names in Golden'].sum()
        TN = totalEntities - (TP + FP + FN)

        confusionMatrixStats.append((modelName, TP, FP, FN, TN))

    totalTP = df['True Positives'].sum()
    totalFP = df['False Positives'].sum()
    totalFN = df['False Negatives'].sum()
    totalEntities = df['Total Names in Generated'].sum() + df['Total Names in Golden'].sum()
    totalTN = totalEntities - (totalTP + totalFP + totalFN)

    confusionMatrixStats.append(('Overall', totalTP, totalFP, totalFN, totalTN))

    return confusionMatrixStats


def displayConfusionMatrixTable(confusionMatrixStats):
    """Display confusion matrix as a Matplotlib table."""
    dfConfusion = pd.DataFrame(confusionMatrixStats, columns=["Model", "TP", "FP", "FN", "TN"])

    fig, ax = plt.subplots(figsize=(10, len(confusionMatrixStats) * 0.5 + 1))
    ax.axis('off')

    plt.title("Confusion Matrix for NER Models", fontsize=14, weight='bold')

    table = ax.table(cellText=dfConfusion.values,
                     colLabels=dfConfusion.columns,
                     cellLoc='center',
                     loc='center',
                     colColours=["#f2f2f2"] * len(dfConfusion.columns))

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)

    plt.show()


def calculateModelAccuracy(comparisonsDf, corpusWordCount):
    """Calculates and displays the accuracy of each model and the overall average accuracy."""
    modelStats = []
    totalCorrectMatches = comparisonsDf['Match'].sum()
    totalWordsInGenerated = comparisonsDf['Total Names in Generated'].sum()

    for modelName, modelDf in comparisonsDf.groupby('Model'):
        modelCorrectMatches = modelDf['Match'].sum()
        modelTotalWords = modelDf['Total Names in Generated'].sum()

        modelAccuracy = round((modelCorrectMatches / modelTotalWords) * 100, 2) if modelTotalWords > 0 else 0
        modelStats.append((modelName, modelCorrectMatches, modelTotalWords, modelAccuracy))

    overallAccuracy = round((totalCorrectMatches / totalWordsInGenerated) * 100, 2) if totalWordsInGenerated > 0 else 0
    modelStats.append(('Overall', totalCorrectMatches, totalWordsInGenerated, overallAccuracy))

    return modelStats, overallAccuracy


def processNerFiles(filePath, goldenSet, nerFiles, columnOfInterest):
    """Processes each NER file, merges with the golden set, and calculates stats."""
    allComparisonsDf = pd.DataFrame()

    for nerFile in nerFiles:
        nerSet = pd.read_csv(os.path.join(filePath, nerFile))

        modelName = nerFile.split('.')[0]

        comparisonDf = createComparisonDataset(goldenSet, nerSet, columnOfInterest, modelName)

        allComparisonsDf = pd.concat([allComparisonsDf, comparisonDf], ignore_index=True)

    return allComparisonsDf


def createComparisonDataset(goldenSet, nerSet, columnOfInterest, modelName):
    """Merges golden set and NER set on 'Grant Number' and prepares the comparison dataset."""
    mergedDf = pd.merge(goldenSet, nerSet, on='Grant Number', suffixes=('_golden', '_ner'))
    comparisons = mergedDf[['Grant Number', columnOfInterest + '_golden', columnOfInterest + '_ner']].copy()
    comparisons.columns = ['Grant Number', 'Golden Names', 'Generated Names']

    comparisons['Golden Names'] = comparisons['Golden Names'].str.replace('[', '').str.replace(']', '')
    comparisons['Generated Names'] = comparisons['Generated Names'].str.replace('[', '').str.replace(']',
                                                                                                     '').str.replace(
        "'", "")

    comparisons['Model'] = modelName
    comparisons['Match'] = comparisons.apply(
        lambda row: countSubstringMatches(row['Generated Names'], row['Golden Names']), axis=1)
    comparisons['Match Percentage'] = comparisons.apply(
        lambda row: calculateMatchPercentage(row['Match'], row['Generated Names']), axis=1)
    comparisons['Total Names in Generated'] = comparisons['Generated Names'].apply(
        lambda x: len(x.split(',')) if pd.notna(x) else 0)
    comparisons['Total Names in Golden'] = comparisons['Golden Names'].apply(
        lambda x: len(x.split(',')) if pd.notna(x) else 0)

    return comparisons


def countSubstringMatches(generatedNames, goldenNames):
    if pd.isna(generatedNames) or pd.isna(goldenNames):
        return 0
    generatedList = [name.strip() for name in generatedNames.split(',')]
    matches = sum(1 for genName in generatedList if genName in goldenNames)
    return matches


def calculateMatchPercentage(matches, generatedNames):
    if pd.isna(generatedNames) or matches == 0:
        return 0.0
    totalNames = len([name.strip() for name in generatedNames.split(',')])
    return round((matches / totalNames) * 100, 2)


def displayModelStatistics(modelStats, overallAccuracy):
    print("Model Statistics:")
    print(f"{'Model':<20} {'Correct Matches':<15} {'Total Names':<12} {'Accuracy (%)':<12}")
    print("-" * 60)
    for modelName, correctMatches, totalWords, accuracy in modelStats:
        print(f"{modelName:<20} {correctMatches:<15} {totalWords:<12} {accuracy:<12.2f}")
    print(f"\nOverall accuracy across all models: {overallAccuracy:.2f}%")


def displayModelStatisticsTable(modelStats, overallAccuracy, corpusWordCount):
    dfStats = pd.DataFrame(modelStats, columns=["Model", "Correct Matches", "Total Names", "Accuracy (%)"])
    dfStats.loc[dfStats['Model'] == 'Overall', 'Accuracy (%)'] = overallAccuracy

    fig, ax = plt.subplots(figsize=(8, len(modelStats) * 0.5 + 1))
    ax.axis('off')

    plt.title(f"Total words in corpus: {corpusWordCount}\nModel Statistics", fontsize=14, weight='bold')

    table = ax.table(cellText=dfStats.values,
                     colLabels=dfStats.columns,
                     cellLoc='center',
                     loc='center',
                     colColours=["#f2f2f2"] * len(dfStats.columns))

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.2)

    plt.show()


def calculateTypeIIIMetrics(df):
    """Calculate True Positives, False Positives, and False Negatives for each row, ensuring non-negative values."""
    df['True Positives'] = df['Match']

    df['False Positives'] = df['Total Names in Generated'] - df['True Positives']

    df['False Negatives'] = df.apply(lambda row: max(0, row['Total Names in Golden'] - row['True Positives']), axis=1)


def ensembleVote(row):
    """
    Combines the 'Persons (entities)' predictions from different NER models
    by taking the union of all names.
    """
    personsFlair = set(row['Persons (entities)_flair'].split(', ')) if isinstance(row['Persons (entities)_flair'],
                                                                                  str) else set()
    personsHuggingface = set(row['Persons (entities)_huggingface'].split(', ')) if isinstance(
        row['Persons (entities)_huggingface'], str) else set()
    personsSpacy = set(row['Persons (entities)'].split(', ')) if isinstance(row['Persons (entities)'], str) else set()
    personsStanza = set(row['Persons (entities)_stanza'].split(', ')) if isinstance(row['Persons (entities)_stanza'],
                                                                                    str) else set()

    # Combine all entities from the different models
    allPersons = personsFlair | personsHuggingface | personsSpacy | personsStanza

    # Remove unwanted characters from each person's name
    cleanedPersons = []
    for person in allPersons:
        cleanedPerson = re.sub(r"[\[\]'\"']", "", person).strip()  # Strip [, ], ', " and whitespace
        cleanedPersons.append(cleanedPerson)

    # Return a comma-separated string of all names
    return ', '.join(cleanedPersons)


def createEnsemble(dfFlair, dfHuggingface, dfSpacy, dfStanza):
    """
    Creates an ensemble NER model by combining the results from different NER models
    stored in DataFrames.
    """
    # Merge the DataFrames (using 'Grant Name' and 'Grant Number' as keys)
    mergedDf = dfFlair.merge(dfHuggingface, on=['Grant Name', 'Grant Number'], suffixes=('_flair', '_huggingface'))

    # Merge df_spacy without renaming its columns
    mergedDf = mergedDf.merge(dfSpacy, on=['Grant Name', 'Grant Number'])

    # Merge df_stanza without renaming its columns, but rename the column in df_stanza
    mergedDf = mergedDf.merge(dfStanza.rename(columns={'Persons (entities)': 'Persons (entities)_stanza'}),
                              on=['Grant Name', 'Grant Number'])

    mergedDf['Ensemble_Persons'] = mergedDf.apply(ensembleVote, axis=1)
    return mergedDf


def main():
    dataPath = '../data/'
    corpusFile = os.path.join(dataPath, 'selectedGrants.txt')
    goldenSetFile = os.path.join(dataPath, 'landGrantGoldenSet.csv')

    corpusWordCount = loadCorpus(corpusFile)
    print(f"Total words in corpus: {corpusWordCount}")

    goldenSet = pd.read_csv(goldenSetFile)

    # Load individual model predictions
    dfFlair = pd.read_csv(os.path.join(dataPath, 'landGrantFlair.csv'))
    dfHuggingface = pd.read_csv(os.path.join(dataPath, 'landGrantHuggingFace.csv'))
    dfSpacy = pd.read_csv(os.path.join(dataPath, 'landGrantSpacy.csv'))
    dfStanza = pd.read_csv(os.path.join(dataPath, 'landGrantStanza.csv'))

    # Create the ensemble DataFrame
    dfEnsemble = createEnsemble(dfFlair, dfHuggingface, dfSpacy, dfStanza)

    # Prepare for metric calculation
    nerFiles = [
        'landGrantHuggingFace.csv',
        'landGrantStanza.csv',
        'landGrantSpacy.csv',
        'landGrantFlair.csv'
    ]
    columnOfInterest = 'Persons (entities)'
    allComparisonsDf = processNerFiles(dataPath, goldenSet, nerFiles, columnOfInterest)

    # Add ensemble results to the comparisons
    ensembleComparisonDf = createComparisonDataset(goldenSet, dfEnsemble, columnOfInterest, 'Ensemble')
    allComparisonsDf = pd.concat([allComparisonsDf, ensembleComparisonDf], ignore_index=True)

    modelStats, overallAccuracy = calculateModelAccuracy(allComparisonsDf, corpusWordCount)

    displayModelStatisticsTable(modelStats, overallAccuracy, corpusWordCount)
    plt.show()

    calculateTypeIIIMetrics(allComparisonsDf)

    confusionMatrixStats = calculateConfusionMatrix(allComparisonsDf)
    displayConfusionMatrixTable(confusionMatrixStats)
    plt.show()


if __name__ == "__main__":
    main()