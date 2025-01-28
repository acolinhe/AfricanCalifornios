import pandas as pd
import pickle
from personReader import get_transformed_data
from matchingFunctions import modified_levenshtein_distance, normalize_spanish_names

class Node:
    def __init__(self, record: dict):
        self.name = normalize_spanish_names(record.get("name"))
        self.father = self.sanitize_reference(record.get("father"))
        self.mother = self.sanitize_reference(record.get("mother"))
        self.spouse = self.sanitize_reference(record.get("spouse"))
        self.children = set()
        self.info = record.copy()
        self.info["gender"] = self.normalize_field(record.get("gender"))
        self.info["race"] = self.normalize_field(record.get("race"))
        self.info["age"] = self.sanitize_age(record.get("age"))

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

    def link_relationships(self):
        for node in self.nodes:
            if node.father:
                father_node = self.find_or_create_node(node.father)
                father_node.children.add(node)
                node.father = father_node
            if node.mother:
                mother_node = self.find_or_create_node(node.mother)
                mother_node.children.add(node)
                node.mother = mother_node
            if node.spouse:
                spouse_node = self.find_or_create_node(node.spouse)
                node.info["spouse"] = spouse_node

    def find_or_create_node(self, name: str):
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
    data = get_transformed_data()
    
    trees = []

    for _, records in data.items():
        tree = Tree()
        for record in records:
            tree.add_node(Node(record))
            print(tree)
        tree.link_relationships()
        trees.append(tree)

    return trees

def save_family_trees(trees, filename="family_trees.pkl"):
    with open(filename, "wb") as file:
        pickle.dump(trees, file)

def main():
    family_trees = build_family_trees()
    save_family_trees(family_trees)
    print(f"Family trees saved to 'family_trees.pkl'.")

    for tree in family_trees:
        print("Tree:")
        for node in tree.nodes:
            print(f"Name: {node.name}, Father: {node.father and node.father.name}, Mother: {node.mother and node.mother.name}, Spouse: {node.info.get('spouse')}, Children: {[child.name for child in node.children]}")

if __name__ == "__main__":
    main()
