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

        if nameMatch and descriptionMatch:
            grantName = nameMatch.group(1).strip()
            grantNumber = nameMatch.group(2).strip()
            description = descriptionMatch.group(1).strip()

            landGrants.append({
                "Grant Name": grantName,
                "Grant Number": grantNumber,
                "Description": description
            })
    
    return landGrants

def extractInfo(description):
    doc = nlp(description)
    grantor, grantee, location, year, size = '', '', '', '', ''
    
    for ent in doc.ents:
        if ent.label_ == 'PERSON':
            if not grantor:
                grantor = ent.text
            elif not grantee:
                grantee = ent.text
        elif ent.label_ == 'GPE' or ent.label_ == 'LOC':
            location = ent.text
        elif ent.label_ == 'DATE':
            year = ent.text
    
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
    
    df.to_csv(os.path.join(outputDir, "landGrantSpacySpan.csv"), index=False)
    print(f"CSV file saved: {os.path.join(outputDir, 'landGrantSpacySpan.csv')}")

if __name__ == '__main__':
    main()
