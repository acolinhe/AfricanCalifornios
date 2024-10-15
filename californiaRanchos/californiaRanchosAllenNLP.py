from allennlp.predictors.predictor import Predictor
import allennlp_models.tagging
import pandas as pd
import os
import re

# Load the pretrained AllenNLP NER model
predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/ner-elmo.2021-02-12.tar.gz")


def extract_info_allen(description):
    results = predictor.predict(sentence=description)
    tokens = results['words']
    tags = results['tags']

    grantor, grantee, persons_entities, location, year, size_acres, size_leagues = '', '', [], '', '', '', ''

    for token, tag in zip(tokens, tags):
        if tag == 'U-PERSON' or tag == 'B-PERSON':
            persons_entities.append(token)
            if not grantor:
                grantor = token
            elif not grantee:
                grantee = token
        elif tag == 'U-GPE' or tag == 'B-GPE':
            location = token
        elif tag == 'U-DATE' or tag == 'B-DATE':
            year = token

    # You can also extract other information similarly based on your regexes
    # Extract land sizes
    size_acres_match = re.search(r'(\d+[\.,]?\d*) acres', description)
    if size_acres_match:
        size_acres = size_acres_match.group(1)

    size_leagues_match = re.search(r'(\d+[\.,]?\d*) sq\. leagues', description)
    if size_leagues_match:
        size_leagues = size_leagues_match.group(1)

    return grantor, grantee, persons_entities, year, size_acres, size_leagues


def main():
    filePath = '../data/selectedGrants.txt'
    landGrants = parseGrantsFile(filePath)

    extractedData = []

    for grant in landGrants:
        grantName = grant["Grant Name"]
        grantNumber = grant["Grant Number"]
        description = grant["Description"]
        county = grant["County"]
        coordinates = grant["Coordinates"]

        grantor, grantee, persons_entities, year, size_acres, size_leagues = extract_info_allen(description)

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

    df.to_csv(os.path.join(outputDir, "landGrantAllen.csv"), index=False)
    print(f"CSV file saved: {os.path.join(outputDir, 'landGrantAllen.csv')}")


if __name__ == '__main__':
    main()
