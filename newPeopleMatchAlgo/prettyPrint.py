import json
import csv
from statistics import mean


# Load family trees from file
def load_family_tree(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data
    except Exception as e:
        print(f"Error loading family tree file: {e}")
        return []


# Pretty Print Functionality (for debugging if needed)
def pretty_print_tree(tree):
    """
    Pretty print a family tree to make debugging easier.
    """
    print(json.dumps(tree, indent=4))


# Filter by Date of Birth (Age Checker)
def check_date_of_birth(tree, minimum_age):
    """
    Filters individuals with age >= minimum_age.
    Returns list with name and age of those individuals.
    """
    qualifying_members = []

    def traverse(node):
        if node.get("age") is not None and node["age"] >= minimum_age:
            qualifying_members.append({"name": node["name"], "age": node["age"]})
        for child in node.get("children", []):
            traverse(child)

    for person in tree:
        traverse(person)

    return qualifying_members


# Save Filtered Results to CSV
def save_to_csv(data, file_name, fieldnames):
    """
    Save data into a CSV file.
    """
    try:
        with open(file_name, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        print(f"Data has been saved to {file_name}")
    except Exception as e:
        print(f"Error writing to CSV file: {e}")


# Calculate Metrics
def calculate_metrics(tree):
    """
    Calculate various metrics:
    1. Total number of unique trees.
    2. Average depth of trees.
    3. Shortest tree depth.
    4. Tree with the oldest node.
    5. Average number of children.
    6. Average number of nodes in a tree.
    7. Highest number of nodes in a single tree.
    """
    metrics = {}

    # Total Number of Trees
    metrics["total_trees"] = len(tree)

    # Helper function to recursively calculate tree depth
    def calculate_depth(node):
        if not node.get('children'):
            return 1
        return 1 + max(calculate_depth(child) for child in node.get('children', []))

    # Helper function to calculate total number of nodes in a tree
    def count_nodes(node):
        return 1 + sum(count_nodes(child) for child in node.get('children', []))

    # Depth metrics
    tree_depths = [calculate_depth(person) for person in tree]
    metrics["average_depth"] = mean(tree_depths) if tree_depths else 0
    metrics["shortest_tree"] = min(tree_depths) if tree_depths else 0

    # Node count metrics
    node_counts = [count_nodes(person) for person in tree]
    metrics["average_number_of_nodes"] = mean(node_counts) if node_counts else 0
    metrics["highest_number_of_nodes"] = max(node_counts) if node_counts else 0

    # Find Oldest Node
    def get_oldest_node(node):
        oldest = node
        for child in node.get('children', []):
            candidate = get_oldest_node(child)
            # Ensure the node has a valid age for comparison
            if (candidate.get('age') or 0) > (oldest.get('age') or 0):
                oldest = candidate
        return oldest

    oldest_nodes = [get_oldest_node(person) for person in tree if person]
    valid_oldest_nodes = [node for node in oldest_nodes if node.get('age') is not None]

    if valid_oldest_nodes:
        oldest_node = max(valid_oldest_nodes, key=lambda x: x.get('age') or 0)
    else:
        oldest_node = {"name": "No valid nodes", "age": 0}

    metrics["oldest_node_name"] = oldest_node.get("name", "Unknown")
    metrics["oldest_node_age"] = oldest_node.get("age", 0)

    # Average Number of Children
    def count_children(node):
        return len(node.get("children", [])) + sum(count_children(child) for child in node.get("children", []))

    all_children_counts = [count_children(person) for person in tree]
    metrics["average_number_of_children"] = mean(all_children_counts) if all_children_counts else 0

    return metrics




# Save Metrics to CSV
def save_metrics_to_csv(metrics, file_name):
    """
    Save metrics in CSV format.
    """
    try:
        with open(file_name, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Metric", "Value"])
            for key, value in metrics.items():
                writer.writerow([key, value])
        print(f"Metrics have been saved to {file_name}")
    except Exception as e:
        print(f"Error writing metrics to CSV: {e}")


# Main function
if __name__ == "__main__":
    # Load family tree from JSON file
    file_path = "family_trees.json"  # Replace with your file path if needed
    family_tree = load_family_tree(file_path)

    if not family_tree:
        print("Family tree data could not be loaded. Exiting.")
        exit()

    # Check for individuals with a minimum age
    minimum_age = 18
    print(f"\nFiltering individuals with age >= {minimum_age}...")
    qualifying_members = check_date_of_birth(family_tree, minimum_age)

    # Save filtered members to CSV
    filtered_file_name = "filtered_individuals.csv"
    save_to_csv(
        qualifying_members,
        filtered_file_name,
        fieldnames=["name", "age"]
    )

    # Calculate and display metrics
    print("\nCalculating metrics...")
    metrics = calculate_metrics(family_tree)

    # Save metrics to CSV
    metrics_file_name = "tree_metrics.csv"
    save_metrics_to_csv(metrics, metrics_file_name)
