import pypdf
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline


def extract_pages(pdf_path, start_page, end_page):
    reader = pypdf.PdfReader(pdf_path)
    text = ""

    for page_num in range(start_page - 1, end_page):
        page = reader.pages[page_num]
        text += page.extract_text() + '\n'

    return text


def perform_ner_huggingface(text):
    tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
    model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
    nlp = pipeline("ner", model=model, tokenizer=tokenizer)
    ner_results = nlp(text)
    entities = [(result['word'], result['entity']) for result in ner_results]
    return entities


def main():
    pdf_path = 'california_ranchos.pdf'
    start_page = 13
    end_page = 23

    extracted_text = extract_pages(pdf_path, start_page, end_page)
    print("Extracted Text:\n", extracted_text)

    print("\nNamed Entities using Hugging Face:\n")
    hf_entities = perform_ner_huggingface(extracted_text)
    for entity in hf_entities:
        print(f"{entity[0]} ({entity[1]})")


if __name__ == "__main__":
    main()
