import json


def parse_person_info(person_info, fid, id_counter, name_to_id):
    lines = person_info.strip().split('\n')
    person_dict = {}
    person_dict['id'] = id_counter
    person_dict['fid'] = fid
    person_dict['spouse'] = []
    person_dict['children'] = []
    person_dict['father'] = None
    person_dict['mother'] = None

    if lines:
        person_dict['name'] = lines[0].strip()
        name_to_id[person_dict['name']] = id_counter
        lines = lines[1:]

    for line in lines:
        if line.startswith('Spouse(s):'):
            spouses = line.split(': ', 1)[1].split(', ')
            person_dict['spouse'] = [spouse.strip() for spouse in spouses if spouse.strip() != 'Unknown']
        elif line.startswith('Children:'):
            children = line.split(': ', 1)[1].split(', ')
            person_dict['children'] = [child.strip() for child in children if child.strip() != 'Unknown']
        elif line.startswith('Parents:'):
            parents = line.split(': ', 1)[1].split(' and ')
            if len(parents) == 2:
                person_dict['father'] = parents[0].strip()
                person_dict['mother'] = parents[1].strip()
            elif len(parents) == 1 and parents[0].strip() != 'Unknown':
                person_dict['father'] = parents[0].strip()
        elif ': ' in line:
            key, value = line.split(': ', 1)
            person_dict[key.lower().replace(' ', '_')] = value.strip()
        else:
            print(f"Skipping line: {line}")

    if 'name' in person_dict:
        name_parts = person_dict['name'].split()
        person_dict['name'] = name_parts[0]
        person_dict['lastname'] = ' '.join(name_parts[1:])
    else:
        print(f"Warning: Name not found for entry with id {id_counter}")
        person_dict['name'] = "Unknown"
        person_dict['lastname'] = "Unknown"

    return person_dict


def update_relationships(people, name_to_id):
    for person in people:
        person['spouse'] = [name_to_id[spouse] for spouse in person['spouse'] if spouse in name_to_id]
        person['children'] = [name_to_id[child] for child in person['children'] if child in name_to_id]
        if person['father'] in name_to_id:
            person['father'] = name_to_id[person['father']]
        else:
            person['father'] = None
        if person['mother'] in name_to_id:
            person['mother'] = name_to_id[person['mother']]
        else:
            person['mother'] = None


def create_families(families_info):
    all_familia = []
    people = []
    id_counter = 1
    name_to_id = {}

    for fid, family_info in enumerate(families_info, start=1):
        family_name = f"Familia Pico {fid}"
        all_familia.append({"id": fid, "name": family_name})

        persons_info = family_info.split('\n\n')
        for person_info in persons_info:
            if person_info.strip():
                person_dict = parse_person_info(person_info, fid, id_counter, name_to_id)
                id_counter += 1
                people.append(person_dict)

    update_relationships(people, name_to_id)

    return all_familia, people


with open('familia.txt', 'r') as file:
    data = file.read().strip()

families_info = [data]

all_familia, people = create_families(families_info)

with open('all_familia.json', 'w') as json_file:
    json.dump(all_familia, json_file, indent=4, ensure_ascii=False)

with open('people.json', 'w') as json_file:
    json.dump(people, json_file, indent=4, ensure_ascii=False)
