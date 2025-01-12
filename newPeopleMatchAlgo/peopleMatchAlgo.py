import pandas as pd
from matchingFunctions import modified_levenshtein_distance

class Node:
    def __init__(self, record: dict):
        self.name = record.name
        self.father = None
        self.mother = None
        self.children = set()
        self.info = {}
        for key, value in record.items():
            self.info[key] = value
        
        def __hash__(self):
            return hash(self.name)
        
        # We can update this when needed catch matched people
        def __eq__(self, other):
            return self.name == other.name

        def name(self):
            return self.name
        
        def father(self):
            return self.father
        
        def mother(self):
            return self.mother

        # We can update this when needed to catch matched people
        def merge(self, record):
            for key, value in record.items():
                if key not in self.info:
                    self.info[key] = value
                else:
                    if key == 'father':
                        self.father = value
                    elif key == 'mother':
                        self.mother = value
                    elif key == 'children':
                        self.children = self.children.union(value)
                    else:
                        if self.info[key] != value:
                            self.info[key] = None

def matchNames(p1: Node, p2: Node):
    return modified_levenshtein_distance(p1.name(), p2.name()) < 1

# This function creates a graph of family trees
def getTrees(ecpp, censusRecords):
    trees = set()

    for _, p in ecpp.iterrows():
        node = Node(p)
        trees.add(node)
    
    newNodes = set()

    for census in censusRecords:
        for _, r in census.iterrows():
            for tree in trees.copy():
                for node in tree.copy():
                    if matchNames(node, Node(r)):
                        node.merge(r)
                    else:
                        new = Node(r)
                        if node.father is not None and matchNames(Node(node.info.get('father')), new):
                            node.children.add(new)
                            new.father = node
                            newNodes.add(new)
                        elif node.mother is not None and matchNames(Node(node.info.get('mother')), new):
                            node.children.add(new)
                            new.mother = node
                            newNodes.add(new)
                        else:
                            trees.add(new)
    while newNodes:
        for census in censusRecords:
            for _, r in census.iterrows():
                for tree in newNodes.copy():
                    for node in tree.copy():
                        if matchNames(node, Node(r)):
                            node.merge(r)
                            newNodes.remove(node)
                            trees.remove(node)
                        else:
                            new = Node(r)
                            if node.father() is not None and matchNames(Node(node.info.get('father')), new):
                                node.children.add(new)
                                new.father = node
                                newNodes.add(new)
                                newNodes.remove(node)
                                trees.remove(node)
                            elif node.mother() is not None and matchNames(Node(node.info.get('mother')), new):
                                node.children.add(new)
                                new.mother = node
                                newNodes.add(new)
                                newNodes.remove(node)
                                trees.remove(node)

    return trees