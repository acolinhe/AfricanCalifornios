import stanza
import pandas as pd
import os
import re

# Download English model for stanza
stanza.download('en')
nlp = stanza.Pipeline('en', processors='tokenize,ner')

def parseGrantsFileWithCSVFormat(filePath):
    with open(filePath, 'r', encoding='utf-8') as file:
        text = file.read()
    
    grantsData = text.split('-' * 80)
    
    landGrants = []
    
    for grant in grantsData:
        nameMatch = re.search(r"Grant Name:\s*(.*?),\s*Grant Number:\s*(\d+)", grant)
        descriptionMatch = re.search(r"Description:\s*(.*)", grant, re.DOTALL)
        
        # Updated regex to capture county (everything after "Description:" and before "Co." or "County")
        countyMatch = re.search(r"Description:\s*(.*?)\s*(Co\.|County)", grant)
        county = countyMatch.group(1).strip() if countyMatch else ''
        
        # Loosened regex for Coordinates (allows optional spaces, variable hyphenation, and zone abbreviations)
        coordinatesMatch = re.search(r"In\s*T\s*[\d\-]+\s*[NS]?\s*,?\s*R\s*[\d\-]+\s*[EW]?\s*,?\s*(SBM|MDM|[\w\.]*)", grant)
        coordinates = coordinatesMatch.group(0).strip() if coordinatesMatch else ''
        
        if nameMatch and descriptionMatch:
            grantName = nameMatch.group(1).strip()
            grantNumber = nameMatch.group(2).strip()
            description = descriptionMatch.group(1).strip()

            landGrants.append({
                "Grant Name": grantName,
                "Grant Number": grantNumber,
                "Description": description,
                "County": county,
                "Coordinates": coordinates
            })
    
    return landGrants

def extractInfoWithCSVFormat(description):
    doc = nlp(description)
    grantor, grantee, location, year, size_acres, size_leagues = '', '', '', '', '', ''
    persons_entities = []
    
    for ent in doc.ents:
        entity = ent.text
        label = ent.type
        
        if label == "PERSON":
            persons_entities.append(entity)
            if not grantor:
                grantor = entity
            elif not grantee:
                grantee = entity
        
        elif label == "LOC" or label == "ORG":
            location = entity
        
        elif label == "DATE":
            year = entity

    size_acres_match = re.search(r'(\d+[\.,]?\d*) acres', description)
    if size_acres_match:
        size_acres = size_acres_match.group(1)
    
    size_leagues_match = re.search(r'(\d+[\.,]?\d*) sq\. leagues', description)
    if size_leagues_match:
        size_leagues = size_leagues_match.group(1)
    
    return grantor, grantee, location, year, size_acres, size_leagues, persons_entities

def mainCSVFormat():
    filePath = '../data/selectedGrants.txt'  # Update this to your correct file path
    landGrants = parseGrantsFileWithCSVFormat(filePath)
    
    extractedData = []

    for grant in landGrants:
        grantName = grant["Grant Name"]
        grantNumber = grant["Grant Number"]
        description = grant["Description"]
        county = grant["County"]
        coordinates = grant["Coordinates"]
        
        grantor, grantee, location, year, size_acres, size_leagues, persons_entities = extractInfoWithCSVFormat(description)
    
        extractedData.append({
            "Grant Name": grantName,
            "Grant Number": grantNumber,
            "County": county,
            "Persons (entities)": persons_entities,
            "Year": year,
            "Land Size (Acres)": size_acres,
            "Land Size (Sq. Leagues)": size_leagues,
            "Coordinates": coordinates,
            "Notes": ""  # Placeholder for 'Notes'
        })
    
    df = pd.DataFrame(extractedData)

    outputDir = "../data/"
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
    
    df.to_csv(os.path.join(outputDir, "landGrantStanza.csv"), index=False)
    print(f"CSV file saved: {os.path.join(outputDir, 'landGrantStanza.csv')}")

if __name__ == '__main__':
    mainCSVFormat()
