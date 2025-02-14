import logging
import json
from concurrent.futures import ProcessPoolExecutor
from personReader import get_transformed_data
from matchingFunctions import modified_levenshtein_distance, normalize_spanish_names

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Node:
    def __init__(self, record: dict):
        self.name = normalize_spanish_names(record.get("name"))
        if not self.name:
            raise ValueError("Node must have a valid name")
        self.father = self.sanitize_reference(record.get("father"))
        self.mother = self.sanitize_reference(record.get("mother"))
        self.spouse = self.sanitize_reference(record.get("spouse"))
        self.children = set()
        self.info = record.copy()
        self.info["gender"] = self.normalize_field(record.get("gender"))
        self.info["race"] = self.normalize_field(record.get("race"))
        self.info["age"] = self.sanitize_age(record.get("age"))
        logging.debug(f"Created node: {self.name}")

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return isinstance(other, Node) and self.name == other.name

    def merge(self, record: dict):
        for key, value in record.items():
            if key not in self.info:
                self.info[key] = value
            elif key == "children":
                self.children.update(value)
            elif self.info[key] != value:
                self.info[key] = None

    @staticmethod
    def sanitize_reference(reference):
        return None if not reference or reference.strip() == "" else normalize_spanish_names(reference)

    @staticmethod
    def sanitize_age(age):
        try:
            return int(age) if age else None
        except (ValueError, TypeError):
            return None

    @staticmethod
    def normalize_field(field):
        return field.lower().strip() if field else None

class Tree:
    def __init__(self):
        self.nodes = set()

    def add_node(self, new_node: Node):
        matched = False
        for node in self.nodes:
            if match_names(node, new_node):
                node.merge(new_node.info)
                matched = True
                break
        if not matched:
            self.nodes.add(new_node)
        logging.debug(f"Added node to tree: {new_node.name}")

    def link_relationships(self):
        pending_relationships = list(self.nodes)

        while pending_relationships:
            node = pending_relationships.pop()
            logging.debug(f"Processing relationships for: {node.name}")

            if node.father:
                father_node = self.find_or_create_node(node.father)
                if father_node:
                    father_node.children.add(node)
                    node.father = father_node

            if node.mother:
                mother_node = self.find_or_create_node(node.mother)
                if mother_node:
                    mother_node.children.add(node)
                    node.mother = mother_node

            if node.spouse:
                spouse_node = self.find_or_create_node(node.spouse)
                if spouse_node:
                    node.spouse = spouse_node

    def find_or_create_node(self, name: str):
        if not name:
            return None
        normalized_name = normalize_spanish_names(name)
        for node in self.nodes:
            if node.name == normalized_name:
                return node
        new_node = Node({"name": name})
        self.nodes.add(new_node)
        return new_node

def match_names(node1: Node, node2: Node, threshold: int = 1):
    return modified_levenshtein_distance(node1.name, node2.name) <= threshold

def build_family_trees():
    """Builds family trees while ensuring proper parent-child linking using 1790_census ChildX fields."""
    data = get_transformed_data()
    trees = []

    with ProcessPoolExecutor() as executor:
        results = executor.map(process_records, data.values())
        for dataset_name, result in zip(data.keys(), results):
            tree = Tree()

            for record in result:
                new_node = Node(record)
                tree.add_node(new_node)

                if dataset_name == "1790_census":
                    for i in range(1, 15):
                        child_name = record.get(f"Child{i}")
                        if child_name:
                            child_node = tree.find_or_create_node(child_name)

                            if child_node not in tree.nodes:  # Ensure child is added to the tree
                                tree.nodes.add(child_node)

                            if new_node.info.get("gender") == "male":
                                child_node.father = new_node
                            else:
                                child_node.mother = new_node

                            new_node.children.add(child_node)

            tree.link_relationships()
            trees.append(tree)

    return trees

def process_records(records):
    logging.info(f"Processing {len(records)} records in worker process")
    return records

def save_family_trees(trees, filename="family_trees.json"):
    """Saves family trees in a nested JSON structure with proper ID references."""

    people_lookup = {}
    name_to_id = {}
    id_counter = 1

    for tree in trees:
        for node in tree.nodes:
            if node.name not in name_to_id:
                name_to_id[node.name] = id_counter
                id_counter += 1

    for tree in trees:
        for node in tree.nodes:
            person = {
                "id": name_to_id[node.name],
                "name": node.name,
                "race": node.info.get("race", ""),
                "gender": node.info.get("gender", ""),
                "age": node.info.get("age", ""),
                "spouse": name_to_id.get(node.spouse.name) if isinstance(node.spouse, Node) else None,
                "children": [name_to_id.get(child.name, None) for child in node.children if isinstance(child, Node)],
                "father": name_to_id.get(node.father.name, None) if isinstance(node.father, Node) else None,
                "mother": name_to_id.get(node.mother.name, None) if isinstance(node.mother, Node) else None
            }
            people_lookup[person["id"]] = person

    root_people = [p for p in people_lookup.values() if not p["father"] and not p["mother"]]

    family_trees = [build_family_tree_structure(person, people_lookup) for person in root_people]

    with open(filename, "w", encoding="utf-8") as file:
        json.dump(family_trees, file, indent=4, ensure_ascii=False)

    logging.info(f"Family trees saved to {filename}")

def build_family_tree_structure(person, people_lookup, visited=None):
    """Recursively build a nested family tree structure using IDs, preventing infinite recursion."""
    if visited is None:
        visited = set()

    if person["id"] in visited:
        logging.warning(f"Circular reference detected for {person['name']} (ID: {person['id']})")
        return {"id": person["id"], "name": person["name"], "circular_reference": True}

    visited.add(person["id"])

    return {
        "id": person["id"],
        "name": person["name"],
        "race": person.get("race", ""),
        "gender": person.get("gender", ""),
        "age": person.get("age", ""),
        "spouse": people_lookup.get(person["spouse"], {}).get("id") if person["spouse"] else None,
        "children": [
            build_family_tree_structure(people_lookup[child_id], people_lookup, visited)
            for child_id in person["children"]
            if child_id in people_lookup
        ]
    }

def main():
    try:
        logging.info("Starting family tree construction.")
        family_trees = build_family_trees()
        save_family_trees(family_trees)
        logging.info("Family tree construction complete.")
    except Exception as e:
        logging.error(f"ERROR: {e}", exc_info=True)

if __name__ == "__main__":
   main()

