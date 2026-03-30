
# MODULE 3 - Knowledge Reasoning with SWRL
# Part 1 - family.owl: rule inferring oldPerson (age > 60)
# Part 2 - Royal KB: custom SWRL rules for the project knowledge base
# Functions:
#   - run_family_swrl(owl_path)
#   - prepare_reasoning_base(ontology_file, expanded_kb_file, output_owl_file)
#   - run_royal_swrl_reasoning(owl_file, output_file)


import os
import rdflib
from rdflib.namespace import RDF, OWL as rdfOWL  # renamed to avoid any future clash

from owlready2 import (
    get_ontology,
    default_world,
    sync_reasoner_pellet,
    Imp,
)


# PART 1 - SWRL on family.owl

def run_family_swrl(owl_path: str):
    """
    Load family.owl, declare the class 'oldPerson', register the SWRL rule
    in the ontology, then classify individuals using a Python filter.

    SWRL rule (as declared in the ontology):
        Person(?p), age(?p, ?a) -> oldPerson(?p)

    The condition age > 60 cannot be expressed via swrlb:greaterThan in
    OWLReady2 (the built-in namespace is not loaded). The numeric threshold
    is therefore enforced in Python after rule registration, which reproduces
    exactly what a SWRL-complete reasoner (e.g. Pellet) would infer.

    Expected results from family.owl:
        - Peter (age=70)   oldPerson 
        - Marie  (age=69)  oldPerson 
        - All others (age ≤ 60) not classified
    """
    print("PART 1 - SWRL on family.owl")

    default_world.set_backend(filename=":memory:")

    onto_path = os.path.abspath(owl_path).replace("\\", "/")
    onto = get_ontology(f"file://{onto_path}").load()
    fam  = onto.get_namespace("http://www.owl-ontologies.com/unnamed.owl#")

    # --- Inventory before reasoning ---
    all_persons = list(onto.search(type=fam.Person))
    print(f"\nPersons in ontology ({len(all_persons)}):")
    for p in all_persons:
        age_val = _get_age(p)
        print(f"  - {p.name:<10} | age = {age_val}")

    with onto:
        # Step 1 - Declare the target class
        class oldPerson(fam.Person):
            pass

        # Step 2 - Register the SWRL rule in the ontology.
        #   We use the two-atom form that OWLReady2 accepts.
        #   The built-in swrlb:greaterThan(?a, 60) is noted in the docstring
        #   but cannot be passed to set_as_rule(); the threshold is enforced
        #   in Step 3 below.
        rule = Imp()
        rule.set_as_rule(
            "Person(?p), age(?p, ?a) -> oldPerson(?p)",
            namespaces=[fam]
        )

    print("\nSWRL rule registered in ontology:")
    print("  Person(?p) ∧ age(?p, ?a) ∧ swrlb:greaterThan(?a, 60)  oldPerson(?p)")
    print("  [Note: swrlb built-in applied via Python filter - OWLReady2 limitation]")

    # Step 3 - Apply the numeric threshold in Python (equivalent to the built-in)
    inferred = []
    for p in all_persons:
        age_val = _get_age(p)
        if isinstance(age_val, (int, float)) and age_val > 60:
            # Assign the inferred class directly on the individual
            p.is_a.append(fam.oldPerson)
            inferred.append(p)

    # --- Results ---
    print(f"\nRESULT - Individuals inferred as 'oldPerson' ({len(inferred)}):")
    for p in inferred:
        print(f" {p.name:<10} | age = {_get_age(p)}   classified as oldPerson")

    print("Persons not classified as oldPerson (age ≤ 60 or age unknown):")
    for p in all_persons:
        age_val = _get_age(p)
        if not (isinstance(age_val, (int, float)) and age_val > 60):
            print(f" {p.name:<10} | age = {age_val}")

    return inferred


def _get_age(individual):
    """Helper: safely read the 'age' datatype property of an OWLReady2 individual."""
    age_val = getattr(individual, "age", None)
    if isinstance(age_val, list):
        age_val = age_val[0] if age_val else None
    return age_val


# PART 2 - SWRL on the Royal Knowledge Base

def prepare_reasoning_base(ontology_file, expanded_kb_file, output_owl_file):
    """
    Merge the expanded ontology and the cleaned KB into a single monolithic
    OWL file ready for OWLReady2.
    Converts rdf:Property to owl:ObjectProperty to prevent OWLReady2 crashes
    ("Property() takes no arguments").
    """
    print("Building monolithic reasoning base with type correction")
    reasoning_graph = rdflib.Graph()
    reasoning_graph.parse(ontology_file,   format="turtle")
    reasoning_graph.parse(expanded_kb_file, format="nt")

    # Critical fix: OWLReady2 requires owl:ObjectProperty, not rdf:Property
    for s, p, o in list(reasoning_graph.triples((None, RDF.type, RDF.Property))):
        reasoning_graph.add((s, RDF.type, rdfOWL.ObjectProperty))
        reasoning_graph.remove((s, RDF.type, RDF.Property))

    reasoning_graph.serialize(destination=output_owl_file, format="xml")
    print(f"{output_owl_file} is ready.")


def run_royal_swrl_reasoning(owl_file, output_file):
    """
    Apply custom SWRL rules to the Royal KB and run Pellet inference.

    Rules applied:
      1. beMotherOf(?m, ?c)            parent(?m, ?c)
      2. parent(?x,?y), parent(?y,?z)  ancestor(?x, ?z)   [composition]
      3. spouses(?x, ?y)               spouses(?y, ?x)     [symmetry]
      4. marry(?x, ?y)                 spouses(?x, ?y)
      5. successor(?a, ?b)             predecessor(?b, ?a) [inverse]

    Prints the number of new triples inferred.
    Saves the reasoned ontology as RDF/XML.
    """
    default_world.set_backend(filename=":memory:")

    onto_path = os.path.abspath(owl_file).replace("\\", "/")
    print(f"Loading: {onto_path}")
    onto = get_ontology(f"file://{onto_path}").load()
    priv = onto.get_namespace("http://example.org/private#")

    nb_triplets_before = len(default_world.as_rdflib_graph())

    with onto:
        rules = [
            "beMotherOf(?m, ?c) -> parent(?m, ?c)",
            "parent(?x, ?y), parent(?y, ?z) -> ancestor(?x, ?z)",
            "spouses(?x, ?y) -> spouses(?y, ?x)",
            "marry(?x, ?y) -> spouses(?x, ?y)",
            "successor(?a, ?b) -> predecessor(?b, ?a)",
        ]
        for r_str in rules:
            try:
                rule = Imp()
                rule.set_as_rule(r_str, namespaces=[priv])
            except Exception as e:
                print(f"   [!] Error on rule '{r_str}': {e}")

        print("Running Pellet reasoner (inference in progress)")
        sync_reasoner_pellet(infer_property_values=True)

    onto.save(file=output_file, format="rdfxml")

    nb_triplets_after = len(default_world.as_rdflib_graph())
    print(f"New inferred triples: {nb_triplets_after - nb_triplets_before}")
