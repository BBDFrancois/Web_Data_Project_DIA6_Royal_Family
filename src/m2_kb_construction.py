
# MODULE 2 - KB Construction, Alignment & Expansion

# Functions:
#   - format_entity(text)
#   - format_predicate(text)
#   - is_literal(entity_type)
#   - build_initial_kb(input_csv, output_ttl)
#   - contextual_spotlight_linking(input_csv, output_mapping_csv, output_alignment_ttl)
#   - clean_relation(raw_relation)
#   - fetch_dbpedia_properties_with_signatures()
#   - get_entity_types_with_cache(dbr_uri, cache)
#   - triple_based_predicate_alignment_approach_b(triplets_csv, mapping_csv, output_template_csv)
#   - generate_global_alignment(mapping_entity_csv, output_predicate_csv, output_alignment_ttl)
#   - generate_dynamic_ontology(extracted_csv, aligned_predicates_csv, output_ontology_ttl)
#   - mass_semantic_expansion(initial_kb_file, mapping_entity_csv, output_expanded_nt)
#   - update_schema_and_sanitize_kb()
#   - analyze_graph_health(owl_file)
#   - export_for_kge(owl_file, output_folder)
#   - convert_to_turtle(input_file, output_file)
# ==
#
# IMPORT STRATEGY
#   owlready2 defines names that clash with rdflib: Graph, Namespace, Ontology, etc.
#   Rule: NEVER use "from owlready2 import *".
#   owlready2 symbols are imported explicitly by name only.
#   rdflib symbols are always accessed via explicit imports - they are never overridden.
# ==

import os
import re
import random
import time
from collections import Counter

import pandas as pd
import requests
import torch

# --- rdflib (always explicit, never overridden) ---
import rdflib
from rdflib import Graph, Literal, RDF, URIRef, Namespace
from rdflib.namespace import OWL, RDFS, XSD

from sentence_transformers import SentenceTransformer, util
from owlready2 import (
    get_ontology,
    default_world,
    sync_reasoner_pellet,
    Imp,
)


# ENTITY & PREDICATE FORMATTING

def format_entity(text):
    """
    Convert a string to a clean PascalCase URI.
    Example: 'Queen Victoria' -> ':QueenVictoria'
    """
    text  = str(text)
    clean = re.sub(r'[^a-zA-Z0-9 ]', '', text)
    words = clean.split()
    if not words:
        return ":UnknownEntity"
    return ":" + "".join(w.title() for w in words)


def format_predicate(text):
    """
    Convert a raw relation string to a camelCase predicate URI.
    Example: 'be the daughter of' -> ':beTheDaughterOf'
    """
    text  = str(text)
    clean = re.sub(r'[^a-zA-Z0-9 ]', '', text)
    words = clean.split()
    if not words:
        return ":relation"
    return ":" + words[0].lower() + "".join(w.title() for w in words[1:])


def is_literal(entity_type):
    """
    Return True if the entity type should be treated as a literal value
    (date, number, etc.) rather than as a named entity URI.
    """
    literal_types = {'DATE', 'TIME', 'PERCENT', 'MONEY',
                     'QUANTITY', 'ORDINAL', 'CARDINAL'}
    return entity_type in literal_types


# INITIAL KB CONSTRUCTION

def build_initial_kb(input_csv, output_ttl):
    """
    Build the initial RDF knowledge base from the extracted triplets CSV.
    Writes a Turtle file with the priv: namespace.
    Deduplicates triplets using a Python set.
    """
    df       = pd.read_csv(input_csv)
    triplets = set()
    for _, row in df.iterrows():
        if pd.isna(row['subject']) or pd.isna(row['object']) or pd.isna(row['relation']):
            continue
        subj = format_entity(row['subject'])
        pred = format_predicate(row['relation'])
        if is_literal(row['object_type']):
            safe_literal = str(row['object']).replace('"', "'")
            obj = f'"{safe_literal}"'
        else:
            obj = format_entity(row['object'])
        triplets.add(f"{subj} {pred} {obj} .")
    with open(output_ttl, 'w', encoding='utf-8') as f:
        f.write("@prefix : <http://example.org/private#> .\n\n")
        for t in sorted(triplets):
            f.write(t + "\n")
    print(f"Knowledge base generated: {output_ttl}")
    print(f"Unique triplets: {len(triplets)}")

# ENTITY LINKING (DBpedia Spotlight)

def contextual_spotlight_linking(input_csv, output_mapping_csv, output_alignment_ttl):
    """
    Entity disambiguation using DBpedia Spotlight with the real source sentence
    (column 'context') as context for maximum precision.
    Outputs a mapping CSV and an alignment TTL file with owl:sameAs links.
    """
    print("Starting Entity Linking with DBpedia Spotlight (contextual)")
    df = pd.read_csv(input_csv)
    if 'context' not in df.columns:
        raise ValueError("Column 'context' not found in CSV.")

    entity_mappings = {}
    spotlight_url   = "https://api.dbpedia-spotlight.org/en/annotate"
    headers         = {"Accept": "application/json"}

    for _, row in df.iterrows():
        subj         = str(row['subject']).strip()
        obj          = str(row['object']).strip()
        context_text = str(row['context']).strip()
        if not context_text or context_text == 'nan':
            context_text = f"{subj} {str(row['relation']).strip()} {obj}"

        obj_is_literal = row['object_type'] in [
            'DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL']

        if subj in entity_mappings and (obj_is_literal or obj in entity_mappings):
            continue

        try:
            response = requests.get(spotlight_url, headers=headers,
                                    params={"text": context_text, "confidence": 0.35})
            if response.status_code == 200:
                data = response.json()
                if "Resources" in data:
                    for resource in data["Resources"]:
                        surface_form = resource["@surfaceForm"]
                        dbpedia_uri  = resource["@URI"]
                        confidence   = float(resource["@similarityScore"])
                        if subj.lower() in surface_form.lower() or surface_form.lower() in subj.lower():
                            if subj not in entity_mappings:
                                entity_mappings[subj] = {"uri": f"<{dbpedia_uri}>", "conf": confidence}
                                print(f"[Subject] '{subj}' -> {dbpedia_uri.split('/')[-1]} (Conf: {confidence:.2f})")
                        if not obj_is_literal:
                            if obj.lower() in surface_form.lower() or surface_form.lower() in obj.lower():
                                if obj not in entity_mappings:
                                    entity_mappings[obj] = {"uri": f"<{dbpedia_uri}>", "conf": confidence}
                                    print(f"[Object]  '{obj}' -> {dbpedia_uri.split('/')[-1]} (Conf: {confidence:.2f})")
        except Exception as e:
            print(f"API error for context '{context_text[:30]}': {e}")
        time.sleep(0.3)

    mapping_data       = []
    alignment_triplets = [
        "@prefix : <http://example.org/private#> .",
        "@prefix owl: <http://www.w3.org/2002/07/owl#> .",
        "@prefix dbr: <http://dbpedia.org/resource/> .\n"
    ]

    all_entities = set(df['subject'].dropna()).union(
        set(df[~df['object_type'].isin(
            ['DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL']
        )]['object'].dropna())
    )

    for entity in all_entities:
        private_uri = format_entity(entity)
        if entity in entity_mappings:
            ext_uri = entity_mappings[entity]["uri"]
            conf    = entity_mappings[entity]["conf"]
            mapping_data.append({"Private Entity": private_uri,
                                  "External URI": ext_uri, "Confidence": conf})
            if conf >= 0.5:
                alignment_triplets.append(f"{private_uri} owl:sameAs {ext_uri} .")
        else:
            mapping_data.append({"Private Entity": private_uri,
                                  "External URI": "NOT_FOUND", "Confidence": 0.0})

    pd.DataFrame(mapping_data).to_csv(output_mapping_csv, index=False)
    with open(output_alignment_ttl, 'w', encoding='utf-8') as f:
        f.write("\n".join(alignment_triplets))
    print(f"\nContextual entity linking done. Mapping saved: {output_mapping_csv}")


# PREDICATE ALIGNMENT

def clean_relation(raw_relation):
    """
    Clean a raw relation string by removing stop words and normalising spacing.
    Used before semantic matching against DBpedia properties.
    """
    stop_words = [" in", " on", " at", " of", " to", " her", " his",
                  " their", " a", " the", " s "]
    cleaned = str(raw_relation).lower()
    for word in stop_words:
        cleaned = cleaned.replace(word, " ")
    words = cleaned.split()
    return " ".join(words[:2]) if words else cleaned


def fetch_dbpedia_properties_with_signatures():
    """
    Download DBpedia property schema (labels, domains, ranges) via SPARQL.
    Returns a list of dicts with keys: uri, label, domain, range.
    """
    print("Downloading DBpedia schema (Properties, Domains, Ranges)")
    url     = 'https://dbpedia.org/sparql'
    headers = {'User-Agent': 'UniversityProjectBot/5.0',
               'Accept': 'application/sparql-results+json'}
    query = """
    SELECT DISTINCT ?property ?label ?domain ?range WHERE {
      ?property a rdf:Property .
      ?property rdfs:label ?label .
      OPTIONAL { ?property rdfs:domain ?domain }
      OPTIONAL { ?property rdfs:range ?range }
      FILTER(LANG(?label) = "en")
      FILTER(STRSTARTS(STR(?property), "http://dbpedia.org/ontology/"))
    }
    LIMIT 3000
    """
    properties = []
    try:
        response = requests.get(url, headers=headers,
                                params={'query': query}, timeout=15)
        if response.status_code == 200:
            results = response.json()['results']['bindings']
            for res in results:
                p_uri = res['property']['value']
                properties.append({
                    'uri':    f"dbo:{p_uri.split('/')[-1]}",
                    'label':  res['label']['value'],
                    'domain': res['domain']['value'] if 'domain' in res else None,
                    'range':  res['range']['value']  if 'range'  in res else None
                })
            print(f"{len(properties)} properties loaded from DBpedia schema.")
        else:
            print(f"HTTP error {response.status_code}")
    except Exception as e:
        print(f"Download error: {e}")
    return properties


def get_entity_types_with_cache(dbr_uri, cache):
    """
    Retrieve the full type hierarchy of an entity from DBpedia via a property path query.
    Uses a cache dict to avoid redundant API calls.
    """
    if dbr_uri in cache:
        return cache[dbr_uri]
    url     = 'https://dbpedia.org/sparql'
    headers = {'User-Agent': 'UniversityProjectBot/5.0',
               'Accept': 'application/sparql-results+json'}
    types   = set()
    query   = f"""
    SELECT DISTINCT ?type WHERE {{
      <{dbr_uri}> rdf:type ?directType .
      ?directType rdfs:subClassOf* ?type .
      FILTER(STRSTARTS(STR(?type), "http://dbpedia.org/ontology/"))
    }}
    """
    try:
        response = requests.get(url, headers=headers, params={'query': query})
        if response.status_code == 200:
            for res in response.json()['results']['bindings']:
                types.add(res['type']['value'])
    except Exception as e:
        print(f"Type extraction error for {dbr_uri}: {e}")
    cache[dbr_uri] = types
    time.sleep(0.1)
    return types


def triple_based_predicate_alignment_approach_b(triplets_csv, mapping_csv, output_template_csv):
    """
    Align private predicates to DBpedia properties using semantic soft scoring.
    For each triplet (s, p, o) where both entities are aligned:
      1. Encode p with sentence-transformers
      2. Find top-50 DBpedia properties by cosine similarity
      3. Apply domain/range penalty to produce a final score
      4. Output a CSV template with 5 ranked candidates per relation.
    """
    print("Starting Predicate Alignment (semantic soft scoring approach)")

    device          = "cuda" if torch.cuda.is_available() else "cpu"
    model           = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    dbpedia_props   = fetch_dbpedia_properties_with_signatures()
    if not dbpedia_props:
        return

    corpus_labels      = [p['label'] for p in dbpedia_props]
    corpus_embeddings  = model.encode(corpus_labels, convert_to_tensor=True)
    df_triplets        = pd.read_csv(triplets_csv)
    df_mapping         = pd.read_csv(mapping_csv)

    uri_to_ext = {}
    for _, row in df_mapping.iterrows():
        if row['External URI'] != "NOT_FOUND":
            uri_to_ext[str(row['Private Entity']).strip()] = str(row['External URI']).strip()

    entity_types_cache  = {}
    alignment_candidates = []

    for _, row in df_triplets.iterrows():
        private_s    = f":{row['subject'].replace(' ', '')}"
        private_o    = f":{row['object'].replace(' ', '')}"
        raw_relation = str(row['relation']).strip()
        cleaned_rel  = clean_relation(raw_relation)

        if private_s in uri_to_ext and private_o in uri_to_ext:
            dbr_s   = uri_to_ext[private_s]
            dbr_o   = uri_to_ext[private_o]
            s_types = get_entity_types_with_cache(dbr_s, entity_types_cache)
            o_types = get_entity_types_with_cache(dbr_o, entity_types_cache)

            query_embedding  = model.encode(cleaned_rel, convert_to_tensor=True)
            hits             = util.semantic_search(query_embedding, corpus_embeddings, top_k=50)[0]
            scored_candidates = []

            for hit in hits:
                prop       = dbpedia_props[hit['corpus_id']]
                base_score = hit['score']
                penalty    = 1.0
                if prop['domain'] and s_types and prop['domain'] not in s_types:
                    penalty *= 0.85
                if prop['range']  and o_types and prop['range']  not in o_types:
                    penalty *= 0.85
                scored_candidates.append({
                    'uri': prop['uri'], 'label': prop['label'],
                    'final_score': base_score * penalty
                })

            scored_candidates.sort(key=lambda x: x['final_score'], reverse=True)
            ranked = [f"{c['uri']} ({c['label']}) [Score: {c['final_score']:.2f}]"
                      for c in scored_candidates[:5]]
            while len(ranked) < 5:
                ranked.append("")

            alignment_candidates.append({
                "Relation_Brute": raw_relation,
                "Context_Subject": dbr_s, "Context_Object": dbr_o,
                "Candidat_1": ranked[0], "Candidat_2": ranked[1],
                "Candidat_3": ranked[2], "Candidat_4": ranked[3],
                "Candidat_5": ranked[4]
            })
            print(f"[{cleaned_rel}] -> {ranked[0]}")

    if alignment_candidates:
        pd.DataFrame(alignment_candidates).drop_duplicates(
            subset=['Relation_Brute']).to_csv(output_template_csv, index=False)
        print(f"\nTemplate saved: {output_template_csv}")


def generate_global_alignment(mapping_entity_csv, output_predicate_csv, output_alignment_ttl):
    """
    Build the global alignment graph (entities + predicates).
    - Entities: owl:sameAs links (confidence >= 0.5)
    - Predicates: owl:equivalentProperty (score >= 0.55) or rdfs:subPropertyOf (score >= 0.4)
    Outputs a Turtle alignment file.
    """
    print("Building global alignment graph (Entities + Predicates)")

    PRIV = Namespace("http://example.org/private#")
    DBO  = Namespace("http://dbpedia.org/ontology/")
    DBR  = Namespace("http://dbpedia.org/resource/")

    alignment_graph = Graph()
    alignment_graph.bind("priv", PRIV)
    alignment_graph.bind("dbo",  DBO)
    alignment_graph.bind("dbr",  DBR)
    alignment_graph.bind("owl",  OWL)

    # 1. Entities
    df_entities   = pd.read_csv(mapping_entity_csv)
    entity_count  = 0
    for _, row in df_entities.iterrows():
        priv_raw = str(row['Private Entity']).strip()
        ext_raw  = str(row['External URI']).strip()
        conf     = float(row['Confidence'])
        if ext_raw != "NOT_FOUND" and conf >= 0.5:
            alignment_graph.add((PRIV[priv_raw.lstrip(":")],
                                  OWL.sameAs,
                                  URIRef(ext_raw.strip("<>"))))
            entity_count += 1

    # 2. Predicates
    df_predicates   = pd.read_csv(output_predicate_csv)
    predicate_count = 0
    regex_pattern   = re.compile(r"dbo:([a-zA-Z0-9_]+)\s+\(.*\)\s+\[Score:\s+([0-9.]+)\]")
    for _, row in df_predicates.iterrows():
        raw_rel  = str(row['Relation_Brute'])
        candidat = str(row.get('Candidat_1', '')).strip()
        if pd.notna(candidat) and candidat:
            match = regex_pattern.search(candidat)
            if match:
                dbo_id     = match.group(1)
                score      = float(match.group(2))
                clean_rel  = format_predicate(raw_rel).lstrip(":")
                if score >= 0.55:
                    alignment_graph.add((PRIV[clean_rel], OWL.equivalentProperty, DBO[dbo_id]))
                    predicate_count += 1
                elif score >= 0.4:
                    alignment_graph.add((PRIV[clean_rel], RDFS.subPropertyOf, DBO[dbo_id]))
                    predicate_count += 1

    alignment_graph.serialize(destination=output_alignment_ttl, format="turtle")
    print(f"Global alignment complete")
    print(f"- {entity_count} entities aligned (owl:sameAs).")
    print(f"- {predicate_count} predicates aligned (owl:equivalentProperty).")
    print(f"- Output: {output_alignment_ttl}")

# ONTOLOGY GENERATION

def generate_dynamic_ontology(extracted_csv, aligned_predicates_csv, output_ontology_ttl):
    """
    Generate a private OWL ontology from extracted triplets.
    Steps:
      1. Declare classes from SpaCy entity types
      2. Infer domain/range per relation from majority type statistics
      3. Enrich with symmetric/transitive flags fetched from DBpedia
      4. Serialise as Turtle
    """
    print("Generating dynamic private ontology")

    PRIV           = Namespace("http://example.org/private#")
    ontology_graph = Graph()
    ontology_graph.bind("priv", PRIV)
    ontology_graph.bind("owl",  OWL)
    ontology_graph.bind("rdfs", RDFS)

    df_triplets   = pd.read_csv(extracted_csv)
    literal_types = {'DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL'}

    # Step 1: Classes
    all_types = (set(df_triplets['subject_type'].dropna()) |
                 set(df_triplets['object_type'].dropna()))
    for t in all_types:
        if t not in literal_types:
            ontology_graph.add((PRIV[t], RDF.type, OWL.Class))

    # Step 2: Domain / Range from statistics
    relation_domains: dict = {}
    relation_ranges:  dict = {}
    for _, row in df_triplets.iterrows():
        camel_rel = format_predicate(str(row['relation'])).lstrip(":")
        if camel_rel not in relation_domains:
            relation_domains[camel_rel] = []
            relation_ranges[camel_rel]  = []
            ontology_graph.add((PRIV[camel_rel], RDF.type, RDF.Property))
        if pd.notna(row['subject_type']):
            relation_domains[camel_rel].append(str(row['subject_type']))
        if pd.notna(row['object_type']):
            relation_ranges[camel_rel].append(str(row['object_type']))

    for rel, domains in relation_domains.items():
        if domains:
            most_common = Counter(domains).most_common(1)[0][0]
            if most_common not in literal_types:
                ontology_graph.add((PRIV[rel], RDFS.domain, PRIV[most_common]))

    for rel, ranges in relation_ranges.items():
        if ranges:
            most_common = Counter(ranges).most_common(1)[0][0]
            if most_common in literal_types:
                ontology_graph.add((PRIV[rel], RDFS.range, XSD.string))
            else:
                ontology_graph.add((PRIV[rel], RDFS.range, PRIV[most_common]))

    # Step 3: Symmetric / Transitive flags from DBpedia
    df_aligned    = pd.read_csv(aligned_predicates_csv)
    regex_pattern = re.compile(r"dbo:([a-zA-Z0-9_]+)")
    dbo_to_priv: dict = {}
    for _, row in df_aligned.iterrows():
        candidat = str(row.get('Candidat_1', '')).strip()
        match    = regex_pattern.search(candidat)
        if match:
            dbo_to_priv[match.group(1)] = PRIV[format_predicate(str(row['Relation_Brute'])).lstrip(":")]

    if dbo_to_priv:
        print(f"Fetching advanced logic for {len(dbo_to_priv)} aligned properties")
        dbo_values = " ".join([f"dbo:{pid}" for pid in dbo_to_priv])
        query = f"""
        SELECT ?prop ?type WHERE {{
          VALUES ?prop {{ {dbo_values} }}
          ?prop rdf:type ?type .
          FILTER(?type IN (owl:SymmetricProperty, owl:TransitiveProperty))
        }}
        """
        url     = 'https://dbpedia.org/sparql'
        headers = {'User-Agent': 'UniversityProjectBot/5.0',
                   'Accept': 'application/sparql-results+json'}
        try:
            response = requests.get(url, headers=headers,
                                    params={'query': query}, timeout=15)
            if response.status_code == 200:
                for res in response.json()['results']['bindings']:
                    prop_id      = res['prop']['value'].split('/')[-1]
                    logical_type = res['type']['value'].split('#')[-1]
                    private_uri  = dbo_to_priv.get(prop_id)
                    if private_uri:
                        flag = OWL.SymmetricProperty if logical_type == "SymmetricProperty" \
                               else OWL.TransitiveProperty
                        ontology_graph.add((private_uri, RDF.type, flag))
        except Exception as e:
            print(f"SPARQL error fetching advanced logic: {e}")

    ontology_graph.serialize(destination=output_ontology_ttl, format="turtle")
    print(f"Private ontology generated: {output_ontology_ttl}")


import pandas as pd
import requests
import re
import time
import itertools
import random
from rdflib import Graph, URIRef
from sentence_transformers import SentenceTransformer, util

# KB EXPANSION & DENSIFICATION (HYBRID ANCHORING)

def mass_semantic_expansion(
        initial_kb_file,
        mapping_entity_csv,
        output_expanded_nt,
        densification_sample_ratio=0.2,
        confidence_threshold=0.7,
        similarity_threshold=0.35
):
    """
    Expand the KB by querying DBpedia in two major phases:
      1-3. 2-hop entity-centric expansion (Tree growth) on all confident entities.
      4.   Hybrid anchoring densification (Mesh closure) with parameterized sampling.
    """
    print("Starting massive AI-guided KB expansion & densification")

    kb = Graph()
    kb.parse(initial_kb_file, format="turtle")
    df_entities = pd.read_csv(mapping_entity_csv)

    # 1. Map aligned anchor entities using the parameterized confidence threshold
    aligned_entities = df_entities[
        (df_entities['External URI'] != 'NOT_FOUND') &
        (df_entities['Confidence'] >= confidence_threshold)
        ]

    uris = [str(uri).strip('<>') for uri in aligned_entities['External URI'].tolist()]
    print(f"Loaded {len(uris)} initial URIs passing the {confidence_threshold} confidence threshold.")
    uris_sparql = " ".join([f"<{uri}>" for uri in uris])

    sparql_endpoint = "https://dbpedia.org/sparql"
    headers = {
        'User-Agent': 'ESILV_Project/1.0',
        'Accept': 'application/sparql-results+json'
    }

    # 
    # PHASE 1: Discover available predicates
    # 
    print("Step 1: Mapping available DBpedia relations")
    query_preds = f"""
    SELECT DISTINCT ?p WHERE {{
        VALUES ?entity {{ {uris_sparql} }}
        ?entity ?p ?o .
        FILTER(isURI(?p))
    }}
    """
    response = requests.post(sparql_endpoint, headers=headers, data={'query': query_preds})
    if response.status_code != 200:
        print(f"DBpedia HTTP error {response.status_code}: {response.text}")
        return
    all_predicates = [res['p']['value'] for res in response.json()['results']['bindings']]

    # 
    # PHASE 2: AI semantic filtering
    # 
    print(f"Step 2: Semantic evaluation of {len(all_predicates)} predicates")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    themes = [
        "family kinship parent child spouse marriage relative ancestor",
        "royalty monarch crown reign throne title successor predecessor",
        "residence palace castle birth place death place location"
    ]
    theme_embeddings = model.encode(themes, convert_to_tensor=True)
    dynamic_golden_predicates = []

    for p_uri in all_predicates:
        raw_name = p_uri.split('/')[-1].split('#')[-1]
        clean_name = re.sub('([a-z])([A-Z])', r'\1 \2', raw_name).lower()
        pred_emb = model.encode(clean_name, convert_to_tensor=True)

        # Applying the parameterized similarity threshold
        if any(util.cos_sim(pred_emb, te).item() >= similarity_threshold for te in theme_embeddings):
            dynamic_golden_predicates.append(p_uri)

    preds_sparql = " ".join([f"<{p}>" for p in dynamic_golden_predicates])
    print(
        f"AI selected {len(dynamic_golden_predicates)} highly relevant relations (threshold: {similarity_threshold}).")

    # 
    # PHASE 3: Batch CONSTRUCT (2-hop Expansion)
    # 
    print(f"\nStep 3: Executing 2-hop CONSTRUCT with batching (Tree Growth)")
    batch_size = 30
    total_batches = (len(uris) // batch_size) + 1
    headers['Accept'] = 'text/turtle'

    for i in range(0, len(uris), batch_size):
        batch_uris = uris[i:i + batch_size]
        batch_uris_sparql = " ".join([f"<{u}>" for u in batch_uris])
        current_batch = (i // batch_size) + 1
        print(f"-> Batch {current_batch}/{total_batches} ({len(batch_uris)} entities)")

        # Reverted strictly to 2 hops
        query_construct = f"""
            CONSTRUCT {{
                ?entity ?p1 ?hop1 .
                ?hop1 ?p2 ?hop2 .
            }}
            WHERE {{
                VALUES ?entity {{ {batch_uris_sparql} }}
                VALUES ?p1 {{ {preds_sparql} }}
                ?entity ?p1 ?hop1 .
                OPTIONAL {{
                    VALUES ?p2 {{ {preds_sparql} }}
                    ?hop1 ?p2 ?hop2 .
                    FILTER(isURI(?hop2))
                }}
            }}
            """
        try:
            r = requests.post(sparql_endpoint, headers=headers,
                              data={'query': query_construct}, timeout=60)
            if r.status_code == 200:
                if "Virtuoso" in r.text[:100] or "Error" in r.text[:50]:
                    print(f"   Server internal error on batch {current_batch}.")
                    continue
                temp_graph = Graph()
                try:
                    temp_graph.parse(data=r.text, format="turtle")
                    kb += temp_graph
                    print(f"   Batch {current_batch} OK: {len(temp_graph)} triples added.")
                except Exception:
                    print(f"   Turtle parse error on batch {current_batch}.")
            time.sleep(1)
        except requests.exceptions.RequestException:
            print(f"   Timeout or network error on batch {current_batch}.")

    # 
    # PHASE 4: Double Anchoring (Mesh Densification)
    # 
    print(f"\nStep 4: Densification via Double Anchoring")

    # Extracting ALL DBpedia entities harvested so far
    dbpedia_nodes = set()
    for node in kb.all_nodes():
        if isinstance(node, URIRef) and "dbpedia.org/resource/" in str(node):
            dbpedia_nodes.add(str(node))

    dbpedia_uris = list(dbpedia_nodes)
    print(f"   Harvested {len(dbpedia_uris)} unique DBpedia entities for cross-referencing.")

    chunk_size = 40
    chunks = [dbpedia_uris[i:i + chunk_size] for i in range(0, len(dbpedia_uris), chunk_size)]

    all_chunk_pairs = list(itertools.combinations_with_replacement(chunks, 2))

    # Applying parameterized sampling to the densification phase
    sampled_pair_count = max(1, int(len(all_chunk_pairs) * densification_sample_ratio))
    chunk_pairs = random.sample(all_chunk_pairs, sampled_pair_count)

    total_densify_batches = len(chunk_pairs)
    print(
        f"   Executing {total_densify_batches} cross-reference queries (sampled at {densification_sample_ratio * 100}%)")

    for idx, (chunk_a, chunk_b) in enumerate(chunk_pairs, 1):
        if idx % 10 == 0 or idx == 1:
            print(f"-> Densification Batch {idx}/{total_densify_batches}")

        sparql_uris_a = " ".join([f"<{u}>" for u in chunk_a])
        sparql_uris_b = " ".join([f"<{u}>" for u in chunk_b])

        query_densify = f"""
        CONSTRUCT {{
            ?e1 ?p ?e2 .
        }}
        WHERE {{
            VALUES ?e1 {{ {sparql_uris_a} }}
            VALUES ?e2 {{ {sparql_uris_b} }}
            VALUES ?p {{ {preds_sparql} }}

            ?e1 ?p ?e2 .
            FILTER(?e1 != ?e2)
        }}
        """
        try:
            r = requests.post(sparql_endpoint, headers=headers,
                              data={'query': query_densify}, timeout=60)
            if r.status_code == 200 and "Virtuoso" not in r.text[:100] and "Error" not in r.text[:50]:
                temp_graph = Graph()
                try:
                    temp_graph.parse(data=r.text, format="turtle")
                    if len(temp_graph) > 0:
                        kb += temp_graph
                        print(f"      Found {len(temp_graph)} hidden lateral links!")
                except Exception:
                    pass
            time.sleep(1)  # Crucial pause to avoid IP blacklisting
        except requests.exceptions.RequestException:
            pass  # Silent timeout on densification, proceed to next batch

    # Final save
    kb.serialize(destination=output_expanded_nt, format="nt")
    print(f"\nDone: Graph expanded and densified to {len(kb)} triples, saved to {output_expanded_nt}.")

def update_schema_and_sanitize_kb(expanded_kb_file, ontology_file, alignment_file,
                                   output_clean_kb_file, ontology_expanded_file,
                                   alignment_expanded_file):
    """
    Full cleaning and privatisation pipeline:
      Phase 1 - Remove noisy/malformed triples
      Phase 2 - Absorb and align DBpedia properties into priv: namespace
      Phase 3 - Absorb and align DBpedia entities into priv: namespace
      Phase 4 - Rewrite the entire graph using the private vocabulary
      Phase 5 - Save updated ontology, alignment and clean KB
    """
    print("Starting Architecture Pipeline: Cleaning, Alignment and Privatisation")

    PRIV = Namespace("http://example.org/private#")
    DBO  = Namespace("http://dbpedia.org/ontology/")
    DBR  = Namespace("http://dbpedia.org/resource/")
    DBP  = Namespace("http://dbpedia.org/property/")

    kb    = Graph();  kb.parse(expanded_kb_file, format="nt")
    onto  = Graph();  onto.parse(ontology_file,  format="turtle")
    align = Graph();  align.parse(alignment_file, format="turtle")

    onto.bind("priv", PRIV);  onto.bind("owl", OWL)
    align.bind("priv", PRIV); align.bind("dbo", DBO)
    align.bind("dbr",  DBR);  align.bind("dbp", DBP); align.bind("owl", OWL)

    print(f"-> Initial graph size: {len(kb)} triples.")

    # Phase 1: Noise removal
    print("Step 1/4: Removing noisy literals and malformed URIs")
    triples_to_remove = [
        (s, p, o) for s, p, o in kb
        if (" " in str(s) or " " in str(p) or
            (isinstance(o, URIRef) and " " in str(o)) or
            (isinstance(o, Literal) and len(str(o)) > 150))
    ]
    for t in triples_to_remove:
        kb.remove(t)
    print(f"   {len(triples_to_remove)} noisy triples purged.")

    translation_map: dict = {}

    # Phase 2: Properties
    print("Step 2/4: Absorbing and aligning Properties")
    aligned_props   = {str(o): s for s, p, o in align.triples((None, OWL.equivalentProperty, None))}
    new_props_count = 0
    for s, p, o in kb:
        p_str = str(p)
        if p_str.startswith("http://dbpedia.org/ontology/") or \
           p_str.startswith("http://dbpedia.org/property/"):
            if p_str not in aligned_props:
                prop_name   = p_str.split('/')[-1]
                priv_prop   = PRIV[prop_name]
                is_obj_prop = not isinstance(o, Literal)
                onto.add((priv_prop, RDF.type,
                           OWL.ObjectProperty if is_obj_prop else OWL.DatatypeProperty))
                if not is_obj_prop:
                    onto.add((priv_prop, RDFS.range, RDFS.Literal))
                onto.add((priv_prop, RDFS.domain, OWL.Thing))
                align.add((priv_prop, OWL.equivalentProperty, URIRef(p_str)))
                aligned_props[p_str] = priv_prop
                new_props_count += 1
            translation_map[p_str] = aligned_props[p_str]

    # Phase 3: Entities
    print("Step 3/4: Absorbing and aligning Entities")
    aligned_entities   = {str(o): s for s, p, o in align.triples((None, OWL.sameAs, None))}
    new_entities_count = 0
    for node in set(kb.subjects()) | set(kb.objects()):
        node_str = str(node)
        if isinstance(node, URIRef) and node_str.startswith("http://dbpedia.org/resource/"):
            if node_str not in aligned_entities:
                priv_entity = PRIV[node_str.split('/')[-1]]
                onto.add((priv_entity, RDF.type, OWL.NamedIndividual))
                align.add((priv_entity, OWL.sameAs, URIRef(node_str)))
                aligned_entities[node_str] = priv_entity
                new_entities_count += 1
            translation_map[node_str] = aligned_entities[node_str]

    print(f"   {new_props_count} new properties and {new_entities_count} new entities added.")

    # Phase 4: Privatisation
    print("Step 4/4: Rewriting the entire graph in private vocabulary")
    clean_kb = Graph()
    for s, p, o in kb:
        new_s = translation_map.get(str(s), s)
        new_p = translation_map.get(str(p), p)
        new_o = translation_map.get(str(o), o) if isinstance(o, URIRef) else o
        clean_kb.add((new_s, new_p, new_o))

    # Phase 5: Save
    onto.serialize(destination=ontology_expanded_file,  format="turtle")
    align.serialize(destination=alignment_expanded_file, format="turtle")
    clean_kb.serialize(destination=output_clean_kb_file, format="nt")

    print("\nPipeline complete!")
    print(f"-> Final clean graph: {len(clean_kb)} triples.")
    print(f"-> Updated: {ontology_expanded_file}, {alignment_expanded_file}")
    print(f"-> New fact graph: {output_clean_kb_file}")

# GRAPH STATISTICS

def analyze_graph_health(owl_file):
    """
    Print a health dashboard for the knowledge graph:
    - Total triples, unique entities, unique properties
    - Average degree and connectivity index
    - Class instance distribution (via owlready2)
    """
    print(f"Analysing graph health: {owl_file}")

    # rdflib for metric computation
    g              = rdflib.Graph()
    g.parse(owl_file, format="xml")
    triples_count  = len(g)
    entities       = set(g.subjects()) | set(o for o in g.objects() if isinstance(o, rdflib.URIRef))
    properties     = set(g.predicates())
    avg_degree         = triples_count / len(entities) if entities else 0
    connectivity_index = triples_count / (len(entities) * len(properties)) if properties else 0

    # owlready2 for class distribution (using explicitly imported get_ontology)
    owl2_onto  = get_ontology(f"file://{os.path.abspath(owl_file)}").load()
    class_stats = [
        {"Class": cls.name, "Instances": len(list(cls.instances()))}
        for cls in owl2_onto.classes()
        if len(list(cls.instances())) > 0
    ]
    df_classes = pd.DataFrame(class_stats).sort_values(by="Instances", ascending=False)

    print("SEMANTIC ARCHITECTURE DASHBOARD")
    print(f" {'METRIC':<25} | {'VALUE':<15}")
    print(f" {'Total Triples':<25} | {triples_count:<15}")
    print(f" {'Unique Entities':<25} | {len(entities):<15}")
    print(f" {'Relations (Properties)':<25} | {len(properties):<15}")
    print(f" {'Average Degree':<25} | {avg_degree:.2f} rel/entity")
    print(f" {'Connectivity Index':<25} | {(connectivity_index * 100):.4f}%")
    if not df_classes.empty:
        print("\nClass distribution:")
        print(df_classes.to_string(index=False))


# KGE EXPORT

def export_for_kge(owl_file, output_folder="kge_data"):
    """
    Export the reasoned KB as train/valid/test splits for KGE training.
    Reads a RDF/XML .owl file, keeps only URI-URI triples (no literals),
    shuffles and splits 80/10/10.
    Outputs train.txt, valid.txt, test.txt in TSV format (h TAB r TAB t).
    """
    print(f"Preparing KGE export from {owl_file}")
    g        = rdflib.Graph()
    g.parse(owl_file, format="xml")
    triplets = [
        f"{str(s).split('#')[-1]}\t{str(p).split('#')[-1]}\t{str(o).split('#')[-1]}"
        for s, p, o in g
        if isinstance(o, rdflib.URIRef)
    ]
    print(f"-> {len(triplets)} valid triples extracted for KGE.")
    random.shuffle(triplets)
    total     = len(triplets)
    train_end = int(total * 0.8)
    val_end   = int(total * 0.9)

    os.makedirs(output_folder, exist_ok=True)
    for filename, data in [("train.txt", triplets[:train_end]),
                            ("valid.txt", triplets[train_end:val_end]),
                            ("test.txt",  triplets[val_end:])]:
        with open(os.path.join(output_folder, filename), "w", encoding="utf-8") as f:
            f.write("\n".join(data))

    print(f"Export complete in '{output_folder}'!")
    print(f"   - Train: {train_end} triples")
    print(f"   - Valid: {val_end - train_end} triples")
    print(f"   - Test : {total - val_end} triples")


def convert_to_turtle(input_file, output_file):
    """Convert a RDF/XML .owl file to Turtle format."""
    g = rdflib.Graph()
    g.parse(input_file, format="xml")
    g.serialize(destination=output_file, format="turtle")
    print(f"Conversion successful: {output_file} ({len(g)} triples)")
