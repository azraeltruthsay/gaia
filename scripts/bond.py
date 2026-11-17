#!/usr/bin/env python3
"""See or set where GAIA stands with someone (gaia_common/utils/bonds.py).

Run inside gaia-core (where the palace is mounted):
    docker exec gaia-core python /gaia/GAIA_Project/scripts/bond.py show
    docker exec gaia-core python /gaia/GAIA_Project/scripts/bond.py show Nick
    docker exec gaia-core python /gaia/GAIA_Project/scripts/bond.py set Nick regard 1 "Azrael's friend"
    docker exec gaia-core python /gaia/GAIA_Project/scripts/bond.py history Nick trust

Scales: trust, their_trust (-2..3), loyalty, their_loyalty (0..3), regard (-3..3).
A set is recorded as an event by "Azrael" with your reason; earned points restart.
"""
import argparse
import sys

from gaia_common.utils import bonds
from gaia_common.utils.people import PeopleRegistry


def _one(reg, name):
    found = reg.find_by_name(name)
    if len(found) != 1:
        sys.exit(f"{len(found)} people answer to {name!r}; " +
                 ", ".join(f"{p['id']} {p['names']}" for p in found))
    return found[0]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("show"); s.add_argument("name", nargs="?")
    s = sub.add_parser("set"); s.add_argument("name"); s.add_argument("scale"); s.add_argument("level", type=int)
    s.add_argument("why"); s.add_argument("--by", default="Azrael")
    s = sub.add_parser("history"); s.add_argument("name"); s.add_argument("scale", nargs="?", default="")
    a = p.parse_args()
    reg = PeopleRegistry()
    if a.cmd == "show":
        people = [_one(reg, a.name)] if a.name else reg.everyone()
        for person in people:
            st = bonds.standing(person)
            print(f"{person['names']} {person.get('relationship', '')!r}: " +
                  ", ".join(f"{k} {v:+d}" for k, v in st.items()))
            print(f"    she reads: {bonds.bond_line(person, person['names'][-1])}")
    elif a.cmd == "set":
        person = _one(reg, a.name)
        reg.bond_set(person["id"], a.scale, a.level, a.why, a.by)
        print(f"{person['names'][0]}: {bonds.standing(reg.get(person['id']))}")
    else:
        for e in bonds.history(_one(reg, a.name), a.scale, limit=30):
            print(e)


if __name__ == "__main__":
    main()
