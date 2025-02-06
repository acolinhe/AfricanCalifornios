import pickle
import json
import logging
from matchingFunctions import normalize_spanish_names

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class Node:
    def __init__(self, record):
        self.name = normalize_spanish_names(record.get("name"))
        self.father = record.get("father")
        self.mother = record.get("mother")
        self.spouse = record.get("spouse")
        self.children = set()
        self.info = record.copy()

    def to_dict(self):
        """Convert Node object to a JSON serializable dictionary, ensuring no direct object references."""
        return {
            "name": self.name,
            "father": self.father.name if isinstance(self.father, Node) else self.father,
            "mother": self.mother.name if isinstance(self.mother, Node) else self.mother,
            "spouse": self.spouse.name if isinstance(self.spouse, Node) else self.spouse,
            "children": [child.name if isinstance(child, Node) else child for child in self.children],
            "info": self.info
        }

class Tree:
    def __init__(self):
        self.nodes = set()

    def to_dict(self):
        """Convert Tree object to a JSON serializable dictionary."""
        return {"nodes": [node.to_dict() for node in self.nodes]}

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Node):
            return obj.to_dict()
        if isinstance(obj, Tree):
            return obj.to_dict()
        if isinstance(obj, set):
            return list(obj)
        return super().default(obj)

pkl_file_path = "family_trees.pkl"

with open(pkl_file_path, "rb") as file:
    family_trees = pickle.load(file)

json_data = json.dumps([tree.to_dict() for tree in family_trees], indent=4, cls=CustomJSONEncoder)

json_file_path = "family_trees.json"
with open(json_file_path, "w") as json_file:
    json_file.write(json_data)

print(f"JSON output saved to {json_file_path}")
