import json
import re

def clean_brackets(obj):
    if isinstance(obj, dict):
        return {k: clean_brackets(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_brackets(elem) for elem in obj]
    elif isinstance(obj, str):
        # Remove square brackets ONLY if they are around the entire string
        return re.sub(r'^\[(.*)\]$', r'\1', obj)
    else:
        return obj

# Load your JSON file
with open('integrated_family_trees.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Clean the data
cleaned_data = clean_brackets(data)

# Save the cleaned data back to a new file
with open('cleaned_family_trees.json', 'w', encoding='utf-8') as f:
    json.dump(cleaned_data, f, indent=4, ensure_ascii=False)

print("Brackets removed and saved to cleaned_family_trees.json")
