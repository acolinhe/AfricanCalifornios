import os
import re
import pandas as pd
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"../data/land_grant_processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)

# Import models
import spacy
from flair.data import Sentence
from flair.models import SequenceTagger
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import stanza

# Load NER models
logging.info("Loading NER models...")

nlp_spacy = spacy.load("en_core_web_sm")
logging.info("Loaded spaCy model.")

tagger_flair = SequenceTagger.load('ner')
logging.info("Loaded Flair model.")

tokenizer_hf = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
model_hf = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
ner_pipeline_hf = pipeline("ner", model=model_hf, tokenizer=tokenizer_hf, aggregation_strategy="simple")
logging.info("Loaded HuggingFace model.")

stanza.download('en')
nlp_stanza = stanza.Pipeline('en', processors='tokenize,ner')
logging.info("Loaded Stanza model.")

def preprocessDescription(description):
    # Remove occurrences of 'Gov.' followed by a name (e.g., 'Gov. Alvarado')
    cleaned_description = re.sub(r'Gov\.\s+[A-Z][a-z]+\b', '', description)
    return cleaned_description

def parseGrantsFile(filePath):
    with open(filePath, 'r', encoding='utf-8') as file:
        text = file.read()
    
    grantsData = text.split('-' * 80)
    
    landGrants = []
    
    for grant in grantsData:
        nameMatch = re.search(r"Grant Name:\s*(.*?),\s*Grant Number:\s*(\d+)", grant)
        descriptionMatch = re.search(r"Description:\s*(.*)", grant, re.DOTALL)
        
        countyMatch = re.search(r"Description:\s*(.*?)\s*(Co\.|County)", grant)
        county = countyMatch.group(1).strip() if countyMatch else ''
        
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

def extractInfo(description, ner_model, model_type='spacy'):
    description = preprocessDescription(description)
    
    if model_type == 'spacy':
        doc = ner_model(description)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
    elif model_type == 'flair':
        sentence = Sentence(description)
        ner_model.predict(sentence)
        entities = [(entity.text, entity.tag) for entity in sentence.get_spans('ner')]
    elif model_type == 'huggingface':
        entities = [(ent['word'], ent['entity_group']) for ent in ner_model(description)]
    elif model_type == 'stanza':
        doc = ner_model(description)
        entities = [(ent.text, ent.type) for ent in doc.ents]
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    grantor, grantee, location, year, size_acres, size_leagues = '', '', '', '', '', ''
    persons_entities = []

    for text, label in entities:
        if label in ['PERSON', 'PER']:
            persons_entities.append(text)
            if not grantor:
                grantor = text
            elif not grantee:
                grantee = text
        elif label in ['GPE', 'LOC']:
            location = text
        elif label == 'DATE':
            year = text

    size_acres_match = re.search(r'(\d+[\.,]?\d*) acres', description)
    if size_acres_match:
        size_acres = size_acres_match.group(1)
    
    size_leagues_match = re.search(r'(\d+[\.,]?\d*) sq\. leagues', description)
    if size_leagues_match:
        size_leagues = size_leagues_match.group(1)
    
    return grantor, grantee, persons_entities, year, size_acres, size_leagues

def processGrants(filePath, ner_model, outputFileName, model_type='spacy'):
    logging.info(f"Processing grants using {model_type} model...")
    landGrants = parseGrantsFile(filePath)
    
    extractedData = []

    for grant in landGrants:
        grantName = grant["Grant Name"]
        grantNumber = grant["Grant Number"]
        description = grant["Description"]
        county = grant["County"]
        coordinates = grant["Coordinates"]

        grantor, grantee, persons_entities, year, size_acres, size_leagues = extractInfo(description, ner_model, model_type)
    
        extractedData.append({
            "Grant Name": grantName,
            "Grant Number": grantNumber,
            "County": county,
            "Persons (entities)": ', '.join(persons_entities),
            "Year": year,
            "Land Size (Acres)": size_acres,
            "Land Size (Sq. Leagues)": size_leagues,
            "Coordinates": coordinates,
            "Notes": ""
        })
    
    df = pd.DataFrame(extractedData)

    outputDir = "../data/"
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
    
    df.to_csv(os.path.join(outputDir, outputFileName), index=False)
    logging.info(f"CSV file saved: {os.path.join(outputDir, outputFileName)}")

def main():
    filePath = '../data/selectedGrants.txt'
    
    processGrants(filePath, nlp_spacy, 'landGrantSpacy.csv', model_type='spacy')
    processGrants(filePath, tagger_flair, 'landGrantFlair.csv', model_type='flair')
    processGrants(filePath, ner_pipeline_hf, 'landGrantHuggingFace.csv', model_type='huggingface')
    processGrants(filePath, nlp_stanza, 'landGrantStanza.csv', model_type='stanza')

if __name__ == '__main__':
    main()
