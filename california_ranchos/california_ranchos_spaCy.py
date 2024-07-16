import pypdf
import spacy
import re


def extract_pages(pdf_path, start_page, end_page):
    reader = pypdf.PdfReader(pdf_path)
    text = ""

    for page_num in range(start_page - 1, end_page):
        page = reader.pages[page_num]
        text += page.extract_text() + '\n'

    return text


def split_into_counties(text):
    pattern = r'(ALAMEDA COUNTY|[A-Z ]+ COUNTY)'
    county_sections = re.split(pattern, text)

    counties = []
    for i in range(1, len(county_sections), 2):
        county_name = county_sections[i].strip()
        county_text = county_sections[i + 1].strip()
        counties.append((county_name, county_text))

    return counties


def perform_ner(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities


def main():
    pdf_path = 'california_ranchos.pdf'
    start_page = 13
    end_page = 23

    extracted_text = extract_pages(pdf_path, start_page, end_page)
    print("Extracted Text:\n", extracted_text)

    counties = split_into_counties(extracted_text)

    for i, (county_name, county_text) in enumerate(counties):
        print(f"\nCounty {i + 1} ({county_name}):\n")
        print(county_text)
        print("\nNamed Entities:\n")
        entities = perform_ner(county_text)
        for entity in entities:
            print(f"{entity[0]} ({entity[1]})")


if __name__ == "__main__":
    main()
