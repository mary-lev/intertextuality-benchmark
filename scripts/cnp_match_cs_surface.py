"""Church-Slavonic NT surface matcher (thin caller over cnp_surface.surface_match).
floor 7, top-1 (the low floor is what surfaces «Се человек»); terminal_tiebreak resolves the
«Се человек» 6-way score tie to CS Ин 19:5 (the Ecce homo verse, where «се, человек» is terminal)
instead of the canonically-earlier Мф 11:19 — closing the lone c8 gold gap."""
import sys
from cnp_surface import surface_match

if __name__ == "__main__":
    n = surface_match("data/crime_and_punishment/cs_nt.jsonl", "cs_ref", "results/crime_and_punishment/cnp_cs_surface.jsonl", floor=7.0, topk=1,
                      terminal_tiebreak=True)
    print(f"{n} CS surface connections (floor 7, top-1) -> results/crime_and_punishment/cnp_cs_surface.jsonl", file=sys.stderr)
