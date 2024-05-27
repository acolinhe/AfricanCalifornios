class FamilyUnit:
    def __init__(self, fid):
        self.fid = fid
        self.members = []

    def __str__(self):
        return str(self.to_dict())

    def add_member(self, person_unit):
        self.members.append(person_unit)
        person_unit.family_unit = self

    def get_all_members(self):
        return self.members

    def to_dict(self):
        return {
            "fid": self.fid,
            "members": [member.to_dict() for member in self.members]
        }


class PersonUnit:
    def __init__(self, pid, name, sex=None, race=None, ethnicity=None, baptismal_date=None, children=None, spouse=None, parents=None):
        self.pid = pid
        self.name = name
        self.sex = sex
        self.race = race
        self.ethnicity = ethnicity
        self.baptismal_date = baptismal_date
        self.potential_children = children if children else []
        self.potential_spouse = spouse if spouse else None
        self.potential_parents = parents if parents else []

    def add_potential_child(self, child):
        if child not in self.potential_children:
            self.potential_children.append(child)

    def set_potential_spouse(self, spouse):
        self.potential_spouse = spouse

    def add_potential_parent(self, parent):
        if parent not in self.potential_parents:
            self.potential_parents.append(parent)

    def to_dict(self):
        return {
            "pid": self.pid,
            "name": self.name,
            "sex": self.sex,
            "race": self.race,
            "ethnicity": self.ethnicity,
            "baptismal_date": self.baptismal_date,
            "potential_children": [child.to_dict() for child in self.potential_children],
            "potential_spouse": self.potential_spouse.to_dict() if self.potential_spouse else None,
            "potential_parents": [parent.to_dict() for parent in self.potential_parents]
        }

    def __str__(self):
        return str(self.to_dict())