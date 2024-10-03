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
        
        # Updated regex for County (capture everything between "Description:" and first "Co." or "County")
        county_match = re.search(r"Description:\s*(.*?)\s*(Co\.|County)", grant)
        county = county_match.group(1).strip() if county_match else ''
        
        # Loosened regex for Coordinates
        coordinates_match = re.search(r"In\s*T\s*[\d\-]+\s*[NS]?\s*,?\s*R\s*[\d\-]+\s*[EW]?\s*,?\s*(SBM|MDM|[\w\.]*)", grant)
        coordinates = coordinates_match.group(0).strip() if coordinates_match else ''

        if name_match and description_match:
            grantName = name_match.group(1).strip()
            grantNumber = name_match.group(2).strip()
            description = description_match.group(1).strip()

            landGrant.append({
                "Grant Name": grantName,
                "Grant Number": grantNumber,
                "Description": description,
                "County": county,
                "Coordinates": coordinates
            })
    
    return landGrant

def extractInfo(description):
    entities = nerPipeline(description)
    grantor, grantee, location, year, size_acres, size_leagues = '', '', '', '', '', ''
    persons_entities = []
    
    for ent in entities:
        entity = ent["word"]
        label = ent["entity_group"]
        
        if label == "PER":
            persons_entities.append(entity)
            if not grantor:
                grantor = entity
            elif not grantee:
                grantee = entity
        
        elif label == "LOC":
            location = entity
        
        elif label == "DATE":
            year = entity

    # Extract land sizes
    size_acres_match = re.search(r'(\d+[\.,]?\d*) acres', description)
    if size_acres_match:
        size_acres = size_acres_match.group(1)
    
    size_leagues_match = re.search(r'(\d+[\.,]?\d*) sq\. leagues', description)
    if size_leagues_match:
        size_leagues = size_leagues_match.group(1)
    
    return grantor, grantee, persons_entities, year, size_acres, size_leagues

def main():
    filePath = '../data/selectedGrants.txt'  # Update this to your correct file path
    landGrants = parseGrantsFile(filePath)
    
    extractedData = []

    for grant in landGrants:
        grantName = grant["Grant Name"]
        grantNumber = grant["Grant Number"]
        description = grant["Description"]
        county = grant["County"]
        coordinates = grant["Coordinates"]
        
        grantor, grantee, persons_entities, year, size_acres, size_leagues = extractInfo(description)
    
        extractedData.append({
            "Grant Name": grantName,
            "Grant Number": grantNumber,
            "County": county,
            "Persons (entities)": ', '.join(persons_entities),  # Combine all persons detected
            "Year": year,
            "Land Size (Acres)": size_acres,
            "Land Size (Sq. Leagues)": size_leagues,
            "Coordinates": coordinates,
            "Notes": ""  # Empty field for 'Notes'
        })
    
    df = pd.DataFrame(extractedData)

    outputDir = "../data/"
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
    
    df.to_csv(os.path.join(outputDir, "landGrantHuggingFaceSpan.csv"), index=False)
    print(f"CSV file saved: {os.path.join(outputDir, 'landGrantHuggingFaceSpan.csv')}")

if __name__ == '__main__':
    main()
