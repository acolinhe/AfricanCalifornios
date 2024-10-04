import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
import os

def calculateScores(goldenValues, generatedValues):
    truePositive = sum(1 for gv, pv in zip(goldenValues, generatedValues) if gv == pv and gv != "")
    falsePositive = sum(1 for gv, pv in zip(goldenValues, generatedValues) if gv != pv and pv != "")
    falseNegative = sum(1 for gv, pv in zip(goldenValues, generatedValues) if gv != pv and gv != "")
    
    precision = truePositive / (truePositive + falsePositive) if truePositive + falsePositive > 0 else 0
    recall = truePositive / (truePositive + falseNegative) if truePositive + falseNegative > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0
    
    return precision, recall, f1

filePath = '../data/'

goldenSet = pd.read_csv(os.path.join(filePath, "landGrantGoldenSet.csv"))

generatedFiles = [
    "landGrantHuggingFace.csv",
    "landGrantHuggingFaceSpan.csv",
    "landGrantSpacy.csv",
    "landGrantSpacySpan.csv",
    "landGrantStanza.csv",
    "landGrantStanzaSpan.csv"
]

columnsOfInterest = ["Grant Name", "Grant Number", "County", "Persons (entities)", "Year", "Land Size (Acres)", "Land Size (Sq. Leagues)", "Coordinates"]

scores = {}

for generatedFile in generatedFiles:
    generatedSet = pd.read_csv(os.path.join(filePath, generatedFile))
    
    generatedSet = generatedSet[columnsOfInterest]
    goldenValues = goldenSet[columnsOfInterest]
    
    fileScores = {"precision": {}, "recall": {}, "f1": {}}
    
    for column in columnsOfInterest:
        precision, recall, f1 = calculateScores(goldenValues[column], generatedSet[column])
        
        fileScores["precision"][column] = precision
        fileScores["recall"][column] = recall
        fileScores["f1"][column] = f1
    
    scores[generatedFile] = fileScores

for generatedFile, fileScores in scores.items():
    print(f"Scores for {generatedFile}:")
    for column in columnsOfInterest:
        print(f"  {column} - Precision: {fileScores['precision'][column]:.4f}, Recall: {fileScores['recall'][column]:.4f}, F1-Score: {fileScores['f1'][column]:.4f}")
    print()
