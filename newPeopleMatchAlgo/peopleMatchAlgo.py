import pickle
import logging
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
        logging.info(f"Created node: {self.name}")

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other.name

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
        logging.info(f"Added node to tree: {new_node.name}")

    def link_relationships(self):
        pending_relationships = list(self.nodes)
        processed_nodes = set()

        while pending_relationships:
            node = pending_relationships.pop()
            processed_nodes.add(node)
            logging.info(f"Processing relationships for: {node.name}")

            if node.father:
                father_node = self.find_or_create_node(node.father)
                if father_node:
                    if father_node not in self.nodes:
                        pending_relationships.append(father_node)
                        self.nodes.add(father_node)
                    father_node.children.add(node)
                    node.father = father_node
                    logging.info(f"  - Linked father: {father_node.name}")

            if node.mother:
                mother_node = self.find_or_create_node(node.mother)
                if mother_node:
                    if mother_node not in self.nodes:
                        pending_relationships.append(mother_node)
                        self.nodes.add(mother_node)
                    mother_node.children.add(node)
                    node.mother = mother_node
                    logging.info(f"  - Linked mother: {mother_node.name}")

            if node.spouse:
                spouse_node = self.find_or_create_node(node.spouse)
                if spouse_node:
                    if spouse_node not in self.nodes:
                        pending_relationships.append(spouse_node)
                        self.nodes.add(spouse_node)
                    node.info["spouse"] = spouse_node
                    logging.info(f"  - Linked spouse: {spouse_node.name}")

            for child in node.children.copy():
                child_node = self.find_or_create_node(child)
                if child_node:
                    if child_node not in self.nodes:
                        pending_relationships.append(child_node)
                        self.nodes.add(child_node)
                    child_node.father = node if node.info.get("gender") == "male" else child_node.father
                    child_node.mother = node if node.info.get("gender") == "female" else child_node.mother
                    node.children.add(child_node)
                    logging.info(f"  - Linked child: {child_node.name}")

    def find_or_create_node(self, name: str):
        normalized_name = normalize_spanish_names(name)
        if not normalized_name:
            logging.warning("Skipping creation of node with empty name")
            return None
        for node in self.nodes:
            if node.name == normalized_name:
                return node
        try:
            new_node = Node({"name": name})
            self.nodes.add(new_node)
            return new_node
        except ValueError:
            return None

def match_names(node1: Node, node2: Node, threshold: int = 1):
    return modified_levenshtein_distance(node1.name, node2.name) <= threshold

def build_family_trees():
    data = get_transformed_data()
    trees = []
    
    with ProcessPoolExecutor() as executor:
        results = executor.map(process_records, [list(records[1]) for records in data.items()])
        for result in results:
            tree = Tree()
            for record in result:
                tree.add_node(Node(record))
            tree.link_relationships()
            trees.append(tree)
    
    return trees

def process_records(records):
    logging.info("Processing dataset in worker process")
    return records

def save_family_trees(trees, filename="family_trees.pkl"):
    with open(filename, "wb") as file:
        pickle.dump(trees, file)
    logging.info(f"Family trees saved to {filename}")

def main():
    logging.info("Starting family tree construction.")
    family_trees = build_family_trees()
    save_family_trees(family_trees)
    logging.info("Family tree construction complete.")

if __name__ == "__main__":
    main()
