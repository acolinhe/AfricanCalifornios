import logging
import json
from concurrent.futures import ProcessPoolExecutor
from personReader import get_transformed_data
from matchingFunctions import modified_levenshtein_distance, normalize_spanish_names

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Census year mapping to facilitate age calculations
CENSUS_YEARS = {
    "padron_1767": 1767,
    "padron_1781": 1781,
    "padron_1785": 1785,
    "padron_1821": 1821,
    "1790_census": 1790
}

# Define threshold constants
NAME_MATCH_THRESHOLD = 2       # For general name matching
SURNAME_MATCH_THRESHOLD = 2    # For surname matching
SPOUSE_MATCH_THRESHOLD = 2     # For spouse name matching
CENSUS_NAME_THRESHOLD = 3      # For cross-census matching with family/age support

class Node:
    def __init__(self, record: dict, dataset_name=None):
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

        # Track appearances across different censuses
        self.timeline = {}
        if dataset_name and dataset_name in CENSUS_YEARS:
            census_year = CENSUS_YEARS[dataset_name]
            self.timeline[census_year] = {
                "age": self.info["age"],
                "record": record
            }

            # Estimate birth year if age is available
            if self.info["age"] is not None:
                self.info["birth_year"] = census_year - self.info["age"]

        logging.debug(f"Created node: {self.name}")

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return isinstance(other, Node) and self.name == other.name

    def merge(self, record: dict):
        """Merge record information into this node."""
        for key, value in record.items():
            if key not in self.info:
                self.info[key] = value
            elif key == "children":
                self.children.update(value)
            elif self.info[key] != value and key not in ("age", "birth_year"):
                # For conflicting values, keep as list of alternatives
                if isinstance(self.info[key], list):
                    if value not in self.info[key]:
                        self.info[key].append(value)
                else:
                    self.info[key] = [self.info[key], value]

    def merge_timeline(self, other_node, census_year):
        """Merge another node's information as a different point in time."""
        if census_year in self.timeline:
            logging.warning(f"Census year {census_year} already exists for {self.name}")
            return

        self.timeline[census_year] = {
            "age": other_node.info.get("age"),
            "record": other_node.info
        }

        # Update relationships based on new information
        if other_node.spouse and not self.spouse:
            self.spouse = other_node.spouse

        if other_node.father and not self.father:
            self.father = other_node.father

        if other_node.mother and not self.mother:
            self.mother = other_node.mother

        self.children.update(other_node.children)

        # Recalculate birth year as average from all appearances
        birth_years = []
        for year, data in self.timeline.items():
            if data["age"] is not None:
                birth_years.append(year - data["age"])

        if birth_years:
            self.info["birth_year"] = sum(birth_years) / len(birth_years)

    @staticmethod
    def sanitize_reference(reference):
        return None if not reference or reference.strip() == "" else normalize_spanish_names(reference)

    @staticmethod
    def sanitize_age(age):
        try:
            return float(age) if age else None
        except (ValueError, TypeError):
            return None

    @staticmethod
    def normalize_field(field):
        return field.lower().strip() if field else None

    def get_estimated_birth_year(self):
        """Get best estimate of birth year from all census appearances."""
        if "birth_year" in self.info and self.info["birth_year"]:
            return self.info["birth_year"]

        # Try to calculate from padron_1821 which has birth years
        if 1821 in self.timeline and "birth_year" in self.timeline[1821]["record"]:
            return self.timeline[1821]["record"]["birth_year"]

        # Calculate from age in most recent census
        recent_years = sorted(self.timeline.keys(), reverse=True)
        for year in recent_years:
            if self.timeline[year]["age"] is not None:
                return year - self.timeline[year]["age"]

        return None


class Tree:
    def __init__(self):
        self.nodes = set()

    def add_node(self, new_node: Node):
        matched = False
        for node in self.nodes:
            if match_nodes(node, new_node):
                node.merge(new_node.info)
                # Transfer any timeline information
                for year, data in new_node.timeline.items():
                    if year not in node.timeline:
                        node.timeline[year] = data
                matched = True
                break
        if not matched:
            self.nodes.add(new_node)
        logging.debug(f"Added node to tree: {new_node.name}")

    def link_relationships(self, census_year=None):
        pending_relationships = list(self.nodes)

        while pending_relationships:
            node = pending_relationships.pop()
            
            # Check for and fix self-references
            if isinstance(node.father, str) and normalize_spanish_names(node.father) == node.name:
                logging.warning(f"Fixing self-reference for father: {node.name}")
                node.father = None
                
            if isinstance(node.mother, str) and normalize_spanish_names(node.mother) == node.name:
                logging.warning(f"Fixing self-reference for mother: {node.name}")
                node.mother = None
                
            if isinstance(node.spouse, str) and normalize_spanish_names(node.spouse) == node.name:
                logging.warning(f"Fixing self-reference for spouse: {node.name}")
                node.spouse = None
            
            logging.debug(f"Processing relationships for: {node.name}")

            if node.father:
                if isinstance(node.father, Node):
                    father_node = node.father
                else:
                    father_node = self.find_or_create_node(node.father)
                    
                if father_node and validate_parent_child_relationship(father_node, node, census_year):
                    father_node.children.add(node)
                    node.father = father_node
                else:
                    logging.warning(f"Invalid father-child relationship: {father_node.name if father_node else 'Unknown'} -> {node.name}")

            if node.mother:
                if isinstance(node.mother, Node):
                    mother_node = node.mother
                else:
                    mother_node = self.find_or_create_node(node.mother)
                    
                if mother_node and validate_parent_child_relationship(mother_node, node, census_year):
                    mother_node.children.add(node)
                    node.mother = mother_node
                else:
                    logging.warning(f"Invalid mother-child relationship: {mother_node.name if mother_node else 'Unknown'} -> {node.name}")

            if node.spouse:
                if isinstance(node.spouse, Node):
                    spouse_node = node.spouse
                else:
                    spouse_node = self.find_or_create_node(node.spouse)
                    
                if spouse_node and validate_spouse_relationship(node, spouse_node):
                    # Make spouse relationship bidirectional
                    node.spouse = spouse_node
                    spouse_node.spouse = node
                else:
                    logging.warning(f"Invalid spouse relationship: {node.name} <-> {spouse_node.name if spouse_node else 'Unknown'}")

    def find_or_create_node(self, name: str):
        if not name:
            return None
        normalized_name = normalize_spanish_names(name)

        # First try exact name match
        for node in self.nodes:
            if node.name == normalized_name:
                return node

        # Then try fuzzy matching with a threshold
        for node in self.nodes:
            if modified_levenshtein_distance(node.name, normalized_name) <= NAME_MATCH_THRESHOLD:
                return node

        # Create new node if no match
        new_node = Node({"name": name})
        self.nodes.add(new_node)
        return new_node

    def find_matching_nodes(self, node, threshold=2):
        """Find nodes that potentially match the given node."""
        matches = []
        for candidate in self.nodes:
            if candidate is node:
                continue

            name_distance = modified_levenshtein_distance(node.name, candidate.name)
            if name_distance <= threshold:
                matches.append((candidate, name_distance))

        return sorted(matches, key=lambda x: x[1])


def match_nodes(node1: Node, node2: Node, threshold: int = NAME_MATCH_THRESHOLD):
    """
    Determines if two nodes represent the same person based on name and other attributes.
    """
    # Check name similarity
    name_distance = modified_levenshtein_distance(node1.name, node2.name)
    if name_distance > threshold:
        return False

    # If one has birth year and other has age, check if they're compatible
    birth_year1 = node1.get_estimated_birth_year()
    birth_year2 = node2.get_estimated_birth_year()

    if birth_year1 and birth_year2:
        # Allow for some reporting error in birth years
        if abs(birth_year1 - birth_year2) > 5:
            return False

    # Check if they have compatible genders
    gender1 = node1.info.get("gender")
    gender2 = node2.info.get("gender")
    if gender1 and gender2 and gender1 != gender2:
        return False

    # Check if they have compatible races
    race1 = node1.info.get("race")
    race2 = node2.info.get("race")
    if race1 and race2 and race1 != race2 and ";" not in race1 and ";" not in race2:
        # Allow mixed races to match with either component
        return False

    # If we have spouse names, check if they match
    spouse1 = node1.spouse.name if isinstance(node1.spouse, Node) else node1.spouse
    spouse2 = node2.spouse.name if isinstance(node2.spouse, Node) else node2.spouse

    if spouse1 and spouse2:
        spouse_distance = modified_levenshtein_distance(spouse1, spouse2)
        if spouse_distance > CENSUS_NAME_THRESHOLD:
            # Spouses don't match - strong evidence these are different people
            return False

    # If we reach here, there's enough matching evidence
    return True


def validate_parent_child_relationship(parent_node, child_node, census_year=None):
    """
    Validates if a parent-child relationship is temporally possible.

    Args:
        parent_node: The potential parent node
        child_node: The potential child node
        census_year: The year of the census data

    Returns:
        bool: Whether the relationship is valid
    """
    if parent_node.name == child_node.name:
        logging.warning(f"Prevented self-referential relationship for: {parent_node.name}")
        return False

    # Minimum age difference for parenthood
    MIN_PARENT_AGE = 13

    # Get estimated birth years
    parent_birth_year = parent_node.get_estimated_birth_year()
    child_birth_year = child_node.get_estimated_birth_year()

    # If we have both birth years, check the difference
    if parent_birth_year and child_birth_year:
        age_diff = child_birth_year - parent_birth_year
        return MIN_PARENT_AGE <= age_diff <= 60  # Valid parenting age range

    # If we only have census_year and ages
    if census_year:
        parent_age = None
        child_age = None

        if census_year in parent_node.timeline:
            parent_age = parent_node.timeline[census_year]["age"]
        elif parent_node.info.get("age"):
            parent_age = parent_node.info.get("age")

        if census_year in child_node.timeline:
            child_age = child_node.timeline[census_year]["age"]
        elif child_node.info.get("age"):
            child_age = child_node.info.get("age")

        if parent_age and child_age:
            return parent_age >= child_age + MIN_PARENT_AGE

    # If we can't validate, accept with a warning
    logging.debug(f"Cannot validate parent-child relationship: {parent_node.name} -> {child_node.name}")
    return True


def validate_spouse_relationship(node1, node2):
    """
    Validates if a spouse relationship is plausible based on age and gender.
    """
    # Check gender compatibility (opposite genders)
    gender1 = node1.info.get("gender")
    gender2 = node2.info.get("gender")

    if gender1 and gender2:
        if gender1 == gender2:
            logging.warning(f"Same gender spouse relationship: {node1.name} and {node2.name}")
            return False

    # Check age compatibility (within reasonable range)
    birth_year1 = node1.get_estimated_birth_year()
    birth_year2 = node2.get_estimated_birth_year()

    if birth_year1 and birth_year2:
        age_diff = abs(birth_year1 - birth_year2)
        if age_diff > 25:  # Allow up to 25 years age difference
            logging.warning(f"Large age gap ({age_diff} years) in spouse relationship: {node1.name} and {node2.name}")
            return False

    return True


def validate_child_surname(parent_name, child_name):
    """Validate that the child likely has the correct surname inheritance"""
    if not parent_name or not child_name:
        return True
        
    parent_parts = parent_name.split()
    child_parts = child_name.split()
    
    # Simple case - not enough parts to check
    if len(parent_parts) < 2 or len(child_parts) < 2:
        return True
        
    # Get the surname (typically last part)
    parent_surname = parent_parts[-1]
    
    # Check if parent's surname appears anywhere in child's name
    for part in child_parts:
        if modified_levenshtein_distance(part, parent_surname) <= SURNAME_MATCH_THRESHOLD:
            return True
            
    # For Spanish naming conventions, check paternal surname inheritance
    if len(parent_parts) >= 2 and len(child_parts) >= 3:
        # In "FirstName PaternalSurname MaternalSurname" format
        parent_paternal = parent_parts[-2] if len(parent_parts) >= 3 else parent_parts[-1]
        child_paternal = child_parts[-2] if len(child_parts) >= 3 else child_parts[-1]
        
        if modified_levenshtein_distance(parent_paternal, child_paternal) <= SURNAME_MATCH_THRESHOLD:
            return True
    
    logging.warning(f"Suspicious surname pattern: parent={parent_name}, child={child_name}")
    return False


def group_by_household(records):
    """
    Groups records into likely household units based on consecutive records with family relationships.

    Args:
        records: List of record dictionaries

    Returns:
        List of lists, each inner list containing records for one household
    """
    households = []
    current_household = []
    current_surname = None

    for record in records:
        name_parts = record.get("name", "").split()
        if len(name_parts) > 1:
            surname = name_parts[-1]

            # Start new household if:
            # 1. This is a new surname AND
            # 2. This person has no parents AND
            # 3. We already have records in the current household
            if (surname != current_surname and
                not record.get("father") and
                not record.get("mother") and
                current_household):
                households.append(current_household)
                current_household = []
                current_surname = surname

            current_household.append(record)

    # Add the last household
    if current_household:
        households.append(current_household)

    return households


def match_person_across_census(person, other_census_data, years_between):
    """
    Match a person in one census to the same person in another census.

    Args:
        person: Node object from one census
        other_census_data: List of Node objects from another census
        years_between: Years between censuses

    Returns:
        List of (node, score) tuples of potential matches, sorted by score
    """
    matches = []

    for candidate in other_census_data:
        # Skip exact same node
        if person is candidate:
            continue

        # Calculate name similarity
        name_score = modified_levenshtein_distance(person.name, candidate.name)

        # Check if ages align
        age_matches = False
        if person.info.get("age") is not None and candidate.info.get("age") is not None:
            expected_age = person.info["age"] + years_between
            age_diff = abs(expected_age - candidate.info["age"])
            # Allow some variance in reported ages
            age_matches = age_diff <= min(3, years_between/2)

        # Check if family members also match
        family_matches = False
        if isinstance(person.spouse, Node) and isinstance(candidate.spouse, Node):
            spouse_score = modified_levenshtein_distance(person.spouse.name, candidate.spouse.name)
            family_matches = spouse_score <= 2

        # Combine all evidence
        match_score = name_score
        if not age_matches:
            match_score += 5
        if not family_matches and person.spouse and candidate.spouse:
            match_score += 3

        # Threshold for considering a match
        if name_score <= NAME_MATCH_THRESHOLD or (name_score <= CENSUS_NAME_THRESHOLD and (age_matches or family_matches)):
            matches.append((candidate, match_score))

    return sorted(matches, key=lambda x: x[1])


def build_family_trees_by_census():
    data = get_transformed_data()
    trees_by_census = {}

    with ProcessPoolExecutor() as executor:
        futures = {dataset_name: executor.submit(process_records, records)
                  for dataset_name, records in data.items()}

        for dataset_name, future in futures.items():
            result = future.result()
            
            # Filter out records with no name before processing
            result = [record for record in result if record.get("name") and record.get("name").strip()]
            
            tree = Tree()

            if dataset_name == "padron_1821":
                # Group by household for more accurate family linkage
                households = group_by_household(result)
                for household in households:
                    for record in household:
                        new_node = Node(record, dataset_name)
                        tree.add_node(new_node)
            else:
                for record in result:
                    new_node = Node(record, dataset_name)
                    tree.add_node(new_node)

            # Special handling for 1790 census child references
            if dataset_name == "1790_census":
                for record in result:
                    parent_node = None
                    for node in tree.nodes:
                        if node.name == normalize_spanish_names(record.get("name")):
                            parent_node = node
                            break

                    if parent_node:
                        for i in range(1, 15):
                            child_name = record.get(f"Child{i}")
                            if child_name:
                                child_node = tree.find_or_create_node(child_name)

                                if parent_node.info.get("gender") == "male":
                                    child_node.father = parent_node
                                else:
                                    child_node.mother = parent_node

                                parent_node.children.add(child_node)

            # Link relationships within this census
            census_year = CENSUS_YEARS.get(dataset_name)
            tree.link_relationships(census_year)
            trees_by_census[dataset_name] = tree

    return trees_by_census


def integrate_trees(trees_by_census):
    """
    Integrate trees from multiple census years into a consolidated view.
    """
    # Start with the earliest census as base
    census_keys = sorted(trees_by_census.keys(),
                         key=lambda k: CENSUS_YEARS.get(k, 0))

    master_tree = Tree()

    # Add all nodes from earliest census
    earliest_census = census_keys[0]
    for node in trees_by_census[earliest_census].nodes:
        master_tree.add_node(node)

    # For each subsequent census
    for i in range(1, len(census_keys)):
        current_census = census_keys[i]
        prev_census = census_keys[i-1]

        current_year = CENSUS_YEARS.get(current_census)
        prev_year = CENSUS_YEARS.get(prev_census)
        years_between = current_year - prev_year if current_year and prev_year else 0

        current_tree = trees_by_census[current_census]

        # Try to match people across censuses
        for current_node in current_tree.nodes:
            matched = False

            # Try to find this person in the master tree
            matches = []
            for master_node in master_tree.nodes:
                match_score = match_nodes(master_node, current_node, threshold=2)
                if match_score:
                    matches.append((master_node, match_score))

            if matches:
                # Get best match
                best_match = min(matches, key=lambda x: x[1]) if matches else None
                if best_match:
                    matched = True
                    master_node = best_match[0]

                    # Merge the information from current census
                    master_node.merge_timeline(current_node, current_year)

                    # Update relationships
                    if current_node.spouse and not master_node.spouse:
                        # Try to find spouse in master tree
                        spouse_node = None
                        for node in master_tree.nodes:
                            if isinstance(current_node.spouse, Node) and match_nodes(node, current_node.spouse):
                                spouse_node = node
                                break

                        if not spouse_node and isinstance(current_node.spouse, Node):
                            # Add the spouse to master tree
                            master_tree.add_node(current_node.spouse)
                            spouse_node = current_node.spouse

                        if spouse_node:
                            master_node.spouse = spouse_node
                            spouse_node.spouse = master_node

            # If not matched, add as new person
            if not matched:
                master_tree.add_node(current_node)

    # Final relationship linking
    master_tree.link_relationships()

    return master_tree


def process_records(records):
    logging.info(f"Processing {len(records)} records in worker process")
    return records


def save_family_trees(tree, filename="family_trees.json"):
    """Saves consolidated family tree in a nested JSON structure with proper ID references."""
    people_lookup = {}
    name_to_id = {}
    id_counter = 1

    # Assign IDs to each person
    for node in tree.nodes:
        if node.name not in name_to_id:
            name_to_id[node.name] = id_counter
            id_counter += 1

    # Create people lookup with relationship references
    for node in tree.nodes:
        person = {
            "id": name_to_id[node.name],
            "name": node.name,
            "race": node.info.get("race", ""),
            "gender": node.info.get("gender", ""),
            "birth_year": node.get_estimated_birth_year(),
            "timeline": {},
            "spouse": name_to_id.get(node.spouse.name) if isinstance(node.spouse, Node) else None,
            "children": [name_to_id.get(child.name, None) for child in node.children if isinstance(child, Node)],
            "father": name_to_id.get(node.father.name, None) if isinstance(node.father, Node) else None,
            "mother": name_to_id.get(node.mother.name, None) if isinstance(node.mother, Node) else None
        }

        # Add timeline information
        for year, data in node.timeline.items():
            person["timeline"][str(year)] = {
                "age": data["age"],
                "race": data["record"].get("race", "")
            }

        people_lookup[person["id"]] = person

    # Find root people (those without parents)
    root_people = [p for p in people_lookup.values() if not p["father"] and not p["mother"]]

    # Build nested family trees starting with root people
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
        "birth_year": person.get("birth_year", ""),
        "timeline": person.get("timeline", {}),
        "spouse": people_lookup.get(person["spouse"], {}).get("id") if person["spouse"] else None,
        "children": [
            build_family_tree_structure(people_lookup[child_id], people_lookup, visited.copy())
            for child_id in person["children"]
            if child_id in people_lookup
        ]
    }


def main():
    try:
        logging.info("Starting family tree construction.")

        # Build trees for each census year
        logging.info("Building trees for each census year...")
        trees_by_census = build_family_trees_by_census()

        # Integrate trees across census years
        logging.info("Integrating trees across census years...")
        master_tree = integrate_trees(trees_by_census)

        # Save final integrated tree
        logging.info("Saving integrated family tree...")
        save_family_trees(master_tree, "integrated_family_trees.json")

        logging.info("Family tree construction complete.")
    except Exception as e:
        logging.error(f"ERROR: {e}", exc_info=True)


if __name__ == "__main__":
    main()