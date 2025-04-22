import re
import pandas as pd
import spacy
import os
from tqdm import tqdm

def process_california_ranchos(file_path, output_path):
    """Process California land grant documents and extract structured information."""
    # Load NER model
    nlp = spacy.load("en_core_web_sm")
    
    # Read the file
    with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
        content = file.read()
    
    # Find the start of actual land grant entries
    county_match = re.search(r'ALAMEDA COUNTY', content)
    if county_match:
        content = content[county_match.start():]
    
    # Extract land grants by county
    grants = extract_land_grants(content, nlp)
    print(f"Found {len(grants)} land grants")
    
    # Save results
    df = pd.DataFrame(grants)
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")
    return df

def extract_land_grants(content, nlp):
    """Extract individual land grant entries."""
    grants = []
    
    # Split into county sections
    county_pattern = re.compile(r'(?:^|\n)(?:(?:[XVI]+\.\s*)|(?:\s*))?([A-Z][A-Z\s]+COUNTY)', re.MULTILINE)
    county_matches = list(county_pattern.finditer(content))
    
    # Process each county section
    for i, match in enumerate(tqdm(county_matches, desc="Processing counties")):
        county = match.group(1).strip()
        start = match.end()
        end = county_matches[i+1].start() if i+1 < len(county_matches) else len(content)
        section_text = content[start:end].strip()
        
        # Use the new function to identify grant entries
        grant_entries = identify_grant_entries(section_text)
        
        # Process each identified grant entry
        for entry in grant_entries:
            # Basic grant data
            grant = {
                'name': entry['name'],
                'number': entry['number'],
                'county': county,
                'counties_listed': entry['counties'] if entry['counties'] else county
            }
            
            # Extract structured information with regex
            grant.update(extract_grant_details(entry['description']))
            
            # Extract named entities
            grant.update(extract_entities(entry['description'], nlp))
            
            # Extract location and coordinates
            grant.update(extract_location_and_coordinates(entry['description']))
            
            grants.append(grant)
    
    return grants

def identify_grant_entries(section_text):
    """Identify land grant entries based on capitalized name followed by ', #' or '. #' pattern."""
    # Pattern to match capitalized grant names that MUST start after a newline
    grant_pattern = re.compile(r'\n([A-Z][A-Z\s\'\.\(\)-]+(?:\s*\([^)]*\))?)(?:,|\.) (#\d+(?:-[A-Z])?)', re.MULTILINE)
    
    # List of coordinate markers to exclude
    skip_terms = ['MDM', 'MDM.', 'SBM', 'SBM.', 'SDM', 'SDM.', 'MDBM', 'MDBM.', 
                  'SBBM', 'SBBM.', 'G&SR', 'G&SR.', 'HBM', 'HBM.']
    
    # Add newline to beginning to ensure first entry can be matched too
    section_with_newline = '\n' + section_text
    
    # Find all matches in the section
    matches = list(grant_pattern.finditer(section_with_newline))
    
    # Extract the entries
    entries = []
    for i in range(len(matches)):
        name = matches[i].group(1).strip()
        
        # Skip if the name is just a coordinate marker
        if name in skip_terms:
            continue
            
        number = matches[i].group(2).strip()
        
        # Find the end of this entry (start of next entry or end of text)
        start_pos = matches[i].end()
        end_pos = matches[i+1].start() if i+1 < len(matches) else len(section_with_newline)
        
        # Extract the description text
        description = section_with_newline[start_pos:end_pos].strip()
        
        # Extract county from the beginning of the description
        county_match = re.search(r'^,?\s*([^,\.]+?)(?:\s+Co(?:\.|\s+)|County)', description)
        counties = county_match.group(1).strip() if county_match else ''
        
        entries.append({
            'name': name,
            'number': number,
            'counties': counties,
            'description': description
        })
    
    return entries

def extract_grant_details(text):
    """Extract structured details using regex patterns."""
    details = {}
    
    # Extract grant size
    size_match = re.search(r'Grant of\s+([\d\s\/\-\.]+(?:\s*sq\.)?\s*leagues?)?\s+made', text)
    if size_match and size_match.group(1):
        details['grant_size'] = size_match.group(1).strip()
    
    # Extract grant year(s)
    year_matches = re.findall(r'made in (\d{4})', text)
    if year_matches:
        details['grant_year'] = "; ".join(year_matches)
    
    # Extract governors - now directly extracting them here
    gov_matches = re.findall(r'by Gov\.\s+([A-Za-z]+)', text)
    if gov_matches:
        details['governor'] = "; ".join(list(set(gov_matches)))
    
    # Extract patent information
    patent_match = re.search(r'Patent for ([\d\.,]+) acres issued in (\d{4}) to ([^\.]+)', text)
    if patent_match:
        details['patent_acres'] = patent_match.group(1).replace(',', '')
        details['patent_year'] = patent_match.group(2)
        details['patent_recipients'] = patent_match.group(3).strip()
    
    # Extract location coordinates
    location_match = re.search(r'In (T [^\.]+)', text)
    if location_match:
        details['location'] = location_match.group(1)
    
    # Extract alternative names
    alias_match = re.search(r'(?:Also known as|Also called)\s+([^\.]+)', text, re.IGNORECASE)
    if alias_match:
        details['alternative_names'] = alias_match.group(1).strip()
    
    return details

def extract_entities(text, nlp):
    """Extract named entities using spaCy NER."""
    doc = nlp(text)
    
    # First identify governors to exclude them from general persons list
    governor_pattern = re.compile(r'by Gov\.\s+([A-Za-z]+)', re.IGNORECASE)
    governors = governor_pattern.findall(text)
    
    # Better patent extraction
    patent_pattern = re.compile(r'Patent for ([\d\.,]+) acres issued in (\d{4}) to ([^\.]+)', re.DOTALL)
    patent_match = patent_pattern.search(text)
    patent_info = {}
    if patent_match:
        patent_info['patent_acres'] = patent_match.group(1).replace(',', '')
        patent_info['patent_year'] = patent_match.group(2)
        patent_info['patent_recipients'] = patent_match.group(3).strip()
    
    # Improved coordinate patterns to filter out
    coordinate_patterns = [
        r'T \d+(?:-\d+)?[NS]',
        r'R \d+(?:-\d+)?[EW]',
        r'MDM',
        r'SBM',
        r'HBM',
        r'MDBM'
    ]
    
    entities = {
        'persons': [],
        'locations': [],
        'dates': [],
        'organizations': [],
        'grantors': governors,
        'patent_holders': []
    }
    
    # Process entities found by spaCy
    for ent in doc.ents:
        # Exclude governor names from general persons list
        if ent.label_ == 'PERSON':
            is_governor = False
            for governor in governors:
                if governor.lower() in ent.text.lower() or ent.text.lower() in governor.lower():
                    is_governor = True
                    break
                
            if not is_governor:
                entities['persons'].append(ent.text)
                
                # Check if this person is in patent recipients
                if patent_match and ent.text in patent_match.group(3):
                    entities['patent_holders'].append(ent.text)
                    
        elif ent.label_ in ['GPE', 'LOC']:
            # IMPROVED: Filter out coordinate notations from locations
            is_coordinate = False
            for pattern in coordinate_patterns:
                if re.search(pattern, ent.text):
                    is_coordinate = True
                    break
            
            # Also filter out single letters that might be part of coordinate references
            if not is_coordinate and len(ent.text) > 1:
                entities['locations'].append(ent.text)
                
        elif ent.label_ == 'DATE':
            # Filter out coordinate-like dates (e.g., "T 5N")
            if not any(re.match(pattern, ent.text) for pattern in coordinate_patterns):
                entities['dates'].append(ent.text)
                
        elif ent.label_ == 'ORG':
            # Filter out coordinate markers from organizations
            if not any(marker in ent.text for marker in coordinate_patterns):
                entities['organizations'].append(ent.text)
    
    # Add patent information to entities
    entities.update(patent_info)
    
    # Remove duplicates and convert to strings
    for key in entities:
        if isinstance(entities[key], list):
            entities[key] = list(set(entities[key]))
            entities[key] = "; ".join(entities[key]) if entities[key] else ""
    
    return entities

def extract_location_and_coordinates(text):
    """Separate actual location names from coordinate references."""
    location_info = {}
    
    # Extract coordinates (typically at the end of the entry)
    coord_match = re.search(r'In (T [^\.]+)', text)
    if coord_match:
        location_info['coordinates'] = coord_match.group(1).strip()
    
    # Extract actual location references 
    location_pattern = re.compile(r'near ([^,\.]+?)(?:,|\.|on)', re.IGNORECASE)
    loc_matches = location_pattern.findall(text)
    if loc_matches:
        location_info['geographical_references'] = "; ".join(set(loc_matches))
    
    # Extract water features
    water_pattern = re.compile(r'(?:on|along) (?:the)?\s+([A-Z][a-z]+ (?:River|Creek|Lake))', re.IGNORECASE)
    water_matches = water_pattern.findall(text)
    if water_matches:
        location_info['water_features'] = "; ".join(set(water_matches))
    
    return location_info

def main():
    """Main execution function."""
    input_file = "data/californiaRanchos.txt"
    output_file = "data/california_land_grants.csv"
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    try:
        df = process_california_ranchos(input_file, output_file)
        print("\nExtraction completed successfully.")
        print(f"Total land grants extracted: {len(df)}")
    except Exception as e:
        print(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    main()