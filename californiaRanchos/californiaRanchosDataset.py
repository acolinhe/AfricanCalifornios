import random
import os
import re

def readTextFile(path: str) -> str:
    if os.path.exists(path):
        print(f"File found at: {path}\n")
    else:
        print(f"File not found: {path}\n")

    try:
        with open(path, encoding='utf-8') as file:
            return file.read()
    except UnicodeDecodeError:
        with open(path, encoding='ISO-8859-1') as file:
            return file.read()

def cleanGrantName(grant_name: str) -> str:
    cleaned_name = re.sub(r'\s*(MDM\.|SBM\.|MBM\.)$', '', grant_name)
    cleaned_name = cleaned_name.replace('.', '').replace(',', '').strip()
    return cleaned_name

def extractGrants(text: str):
    pattern = re.compile(
        r'^\s*([A-Z][A-Z\s\(\)\.,]*)\s*#(\d+),\s*([^\n]+(?:\n\s+[^\n]+)*)',
        re.MULTILINE
    )
    
    matches = pattern.findall(text)
    
    grants = [
        {
            "Grant Name": cleanGrantName(match[0].strip()),
            "Grant Number": match[1],
            "Description": match[2].strip()
        }
        for match in matches
    ]
    
    return grants

# Will change this up later for final dataset
def saveGrants(grants, output_path):
    with open(output_path, 'w', encoding='utf-8') as file:
        for grant in grants:
            file.write(f"Grant Name: {grant['Grant Name']}, Grant Number: {grant['Grant Number']}\n")
            file.write(f"Description: {grant['Description']}\n")
            file.write("-" * 80 + "\n")


def main():
    filePath = '../data/californiaRanchos.txt'
    outputPath = '../data/selectedGrants.txt'
    
    californiaRanchosText = readTextFile(filePath)
    landGrants = extractGrants(californiaRanchosText)

    random.shuffle(landGrants)
    selectedGrants = landGrants[:50]
    
    saveGrants(selectedGrants, outputPath)



if __name__ == '__main__':
    main()