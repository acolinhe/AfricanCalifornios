import spacy
import pandas as pd
import os
import re

nlp = spacy.load("es_core_news_sm")

def parseGrantsFile(filePath):
    with open(filePath, 'r', encoding='utf-8') as file:
        text = file.read()
    
    grantsData = text.split('-' * 80)
    
    landGrants = []
    
    for grant in grantsData:
        nameMatch = re.search(r"Grant Name:\s*(.*?),\s*Grant Number:\s*(\d+)", grant)
        descriptionMatch = re.search(r"Description:\s*(.*)", grant, re.DOTALL)
        
        # Updated regex to capture County (between "Description:" and first "Co." or "County")
        countyMatch = re.search(r"Description:\s*(.*?)\s*(Co\.|County)", grant)
        county = countyMatch.group(1).strip() if countyMatch else ''
        
        # Loosened regex to capture Coordinates
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

def extractInfo(description):
    doc = nlp(description)
    grantor, grantee, location, year, size_acres, size_leagues = '', '', '', '', '', ''
    persons_entities = []
    
    for ent in doc.ents:
        if ent.label_ == 'PERSON':
            persons_entities.append(ent.text)
            if not grantor:
                grantor = ent.text
            elif not grantee:
                grantee = ent.text
        elif ent.label_ == 'GPE' or ent.label_ == 'LOC':
            location = ent.text
        elif ent.label_ == 'DATE':
            year = ent.text
    
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
    
    df.to_csv(os.path.join(outputDir, "landGrantSpacySpan.csv"), index=False)
    print(f"CSV file saved: {os.path.join(outputDir, 'landGrantSpacySpan.csv')}")

if __name__ == '__main__':
    main()

