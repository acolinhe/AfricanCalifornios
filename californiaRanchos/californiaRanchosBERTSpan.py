import pandas as pd
import re
import os
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

nerPipeline = pipeline("ner", model="dccuchile/bert-base-spanish-wwm-cased", tokenizer="dccuchile/bert-base-spanish-wwm-cased", aggregation_strategy="simple")


def parseGrantsFile(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    
    grantsData = text.split('-' * 80)
    
    landGrant = []
    
    for grant in grantsData:
        name_match = re.search(r"Grant Name:\s*(.*?),\s*Grant Number:\s*(\d+)", grant)
        description_match = re.search(r"Description:\s*(.*)", grant, re.DOTALL)

        if name_match and description_match:
            grantName = name_match.group(1).strip()
            grantNumber = name_match.group(2).strip()
            description = description_match.group(1).strip()

            landGrant.append({
                "Grant Name": grantName,
                "Grant Number": grantNumber,
                "Description": description
            })
    
    return landGrant

def extractInfo(description):
    entities = nerPipeline(description)
    grantor, grantee, location, year, size = '', '', '', '', ''
    
    for ent in entities:
        entity = ent["word"]
        label = ent["entity_group"]
        
        if label == "PER":
            if not grantor:
                grantor = entity
            elif not grantee:
                grantee = entity
        
        elif label == "LOC":
            location = entity
        
        elif label == "DATE":
            year = entity

    sizeMatch = re.search(r'(\d+[\.,]?\d*) (acres|sq\. leagues)', description)
    if sizeMatch:
        size = sizeMatch.group(0)
    
    return grantor, grantee, location, year, size

def main():
    filePath = '../data/selectedGrants.txt'
    landGrants = parseGrantsFile(filePath)
    
    extractedData = []

    for grant in landGrants:
        grantName = grant["Grant Name"]
        grantNumber = grant["Grant Number"]
        description = grant["Description"]
        
        grantor, grantee, location, year, size = extractInfo(description)
    
        extractedData.append({
            "Grant Name": grantName,
            "Grant Number": grantNumber,
            "Location": location,
            "Grantor": grantor,
            "Grantee": grantee,
            "Year": year,
            "Land Size (Acres)": size
        })
    
    df = pd.DataFrame(extractedData)

    outputDir = "../data/"
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
    
    df.to_csv(os.path.join(outputDir, "landGrantHuggingFaceSpan.csv"), index=False)
    print(f"CSV file saved: {os.path.join(outputDir, 'landGrantHuggingFaceSpan.csv')}")

if __name__ == '__main__':
    main()
