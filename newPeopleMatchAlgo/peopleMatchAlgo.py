import pandas as pd
from matchingFunctions import modified_levenshtein_distance, normalize_spanish_names

class Node:
    def __init__(self, record: dict):
        """Initialize a Node with personal information."""
        self.name = normalize_spanish_names(record.get("name", ""))
        self.father = None
        self.mother = None
        self.children = set()
        # Maybe change. This is a dictionary of personal information.
        self.info = record.copy()

    # probably remove
    def __hash__(self):
        """Ensure Node can be used in sets."""
        return hash(self.name)

    def __eq__(self, other):
        """Compare Nodes by name."""
        return self.name == other.name

    def merge(self, record: dict):
        """Merge additional information into the Node."""
        for key, value in record.items():
            if key not in self.info:
                self.info[key] = value
            else:
                if key == "father":
                    self.father = value
                elif key == "mother":
                    self.mother = value
                elif key == "children":
                    self.children.update(value)
                elif self.info[key] != value:
                    self.info[key] = None

def match_names(node1: Node, node2: Node, threshold: int = 1):
    """Check if two Nodes represent the same person using Levenshtein distance."""
    return modified_levenshtein_distance(node1.name, node2.name, custom_costs={}) <= threshold

# Common case where it's just one node -> no family
# Probably need a function called createTree of just 1 node
# then another funciton caleld mergeTree that takes in a tree and a node
# play around with creating a Tree class -> having a tree class could be useful
def create_tree(nodes: set, new_node: Node):
    """Add a new Node to a tree if a match is found."""
    matched = False
    for node in nodes.copy():
        if match_names(node, new_node):
            node.merge(new_node.info)
            matched = True
            break
    if not matched:
        nodes.add(new_node)

def get_trees(ecpp, census_records):
    """Build family trees from ECPP and census data."""
    trees = set()

    # Initialize trees from ECPP data
    # Function to read the doc and process
    # Dictionary at top level, in it is name of document with description, year, then it'll have members that are a list of dictionaries
    for _, record in ecpp.iterrows():
        trees.add(Node(record))

    new_nodes = set()

    # Process census records and build relationships
    for census in census_records:
        for _, record in census.iterrows():
            record_node = Node(record)
            create_tree(trees, record_node)

            for tree in trees.copy():
                for node in tree.copy():
                    if node.father and match_names(Node(node.info.get("father")), record_node):
                        node.children.add(record_node)
                        record_node.father = node
                        new_nodes.add(record_node)
                    elif node.mother and match_names(Node(node.info.get("mother")), record_node):
                        node.children.add(record_node)
                        record_node.mother = node
                        new_nodes.add(record_node)
                    else:
                        trees.add(record_node)

    # Process new nodes iteratively
    while new_nodes:
        for census in census_records:
            for _, record in census.iterrows():
                record_node = Node(record)
                for tree in new_nodes.copy():
                    for node in tree.copy():
                        if match_names(node, record_node):
                            node.merge(record_node.info)
                            new_nodes.remove(node)
                        else:
                            if node.father and match_names(Node(node.info.get("father")), record_node):
                                node.children.add(record_node)
                                record_node.father = node
                                new_nodes.add(record_node)
                                new_nodes.remove(node)
                            elif node.mother and match_names(Node(node.info.get("mother")), record_node):
                                node.children.add(record_node)
                                record_node.mother = node
                                new_nodes.add(record_node)
                                new_nodes.remove(node)
    return trees

def main():
    ecpp = pd.read_csv("ecpp.csv")  # Replace with actual ecpp path
    census_files = [] # Replace with actual census/padrinos paths
    census_records = [pd.read_csv(file) for file in census_files]

    # Build trees
    family_trees = get_trees(ecpp, census_records)

    # Output results
    for tree in family_trees:
        print("Tree:")
        for node in tree:
            print(f"Name: {node.name}, Father: {node.father}, Mother: {node.mother}, Children: {[child.name for child in node.children]}")

if __name__ == "__main__":
    main()
