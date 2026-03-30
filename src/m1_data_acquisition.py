
# MODULE 1 - Data Acquisition & Information Extraction
# Domain: British Royal Family (19th century)
# Functions:
#   - fetching(url)
#   - checker_usefulness(extracted_txt, nb_cara)
#   - save_to_jsonl(url, file, txt, new)
#   - scraping_site(root_url, file, nb_leafs, nb_cara, new)
#   - get_extended_context(token, doc, window)
#   - is_clean_token(token)
#   - get_full_entity(t, doc_ents)
#   - get_complete_chunk(token)
#   - name_entity_recognition(text)
#   - standardize_relation(extracted_relation, threshold)
#   - extract_knowledge(input_file, output_file, similarity_threshold)

import trafilatura
from trafilatura.sitemaps import sitemap_search
import spacy
import requests
import json
import time
import random
import csv
import re
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# ------------------ SCRAPING ----------------------

# --- CONFIGURATION ---
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept-Language': 'en-US,en;q=0.9'
}


def fetching(url):
    """
    Download the main text for a given URL.
    Uses trafilatura to isolate the main content of the page.
    """
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        return trafilatura.extract(
            response.text,
            include_comments=False,
            include_tables=False,
            no_fallback=True,
            output_format='txt'
        )
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None


def checker_usefulness(extracted_txt, nb_cara):
    """
    Check whether extracted text is long enough to be useful.
    Returns True if text length >= nb_cara characters.
    """
    if not extracted_txt:
        return False
    return len(extracted_txt) >= nb_cara


def save_to_jsonl(url, file, txt, new=False):
    """
    Save the text of a URL in JSONL format (one line per URL).
    mode 'w' overwrites the file, mode 'a' appends to it.
    """
    clean_txt = " ".join(txt.split())
    entry = {"url": url, "text": clean_txt}
    mode = "w" if new else "a"
    with open(file, mode, encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Saved: {url}")


def scraping_site(root_url, file, nb_leafs, nb_cara=1000, new=True):
    """
    Main scraping function.
    Crawls up to nb_leafs URLs from the sitemap of root_url,
    keeps only pages with at least nb_cara characters,
    and saves them in JSONL format.
    """
    print(f"Starting scraping on: {root_url}")
    leafs_url = sitemap_search(root_url)[:nb_leafs]
    leafs_url.append(root_url)
    print(f"{len(leafs_url)} links found.")
    is_first_write = new
    for url in leafs_url:
        extracted_txt = fetching(url)
        if checker_usefulness(extracted_txt, nb_cara):
            save_to_jsonl(url, file, extracted_txt, new=is_first_write)
            is_first_write = False
            sleep_time = random.uniform(1, 2)
            time.sleep(sleep_time)
    print(f"Output saved in {file}")


# --------------------- NLP UTILITIES ----------------------

nlp = spacy.load("en_core_web_md")


def get_extended_context(token, doc, window=0):
    """
    Retrieve the sentence containing the token.
    If window=1, also retrieve the previous and next sentences.
    """
    if window == 0:
        return token.sent.text.strip()
    sentences = list(doc.sents)
    sent_idx = -1
    for i, sent in enumerate(sentences):
        if sent.start <= token.i < sent.end:
            sent_idx = i
            break
    if sent_idx == -1:
        return token.sent.text.strip()
    start = max(0, sent_idx - window)
    end = min(len(sentences), sent_idx + window + 1)
    return " ".join([s.text.strip() for s in sentences[start:end]])


def is_clean_token(token):
    """
    Filter out pronouns and generic reference words
    that are not useful as entity anchors.
    """
    if token.pos_ == "PRON":
        return False
    BLACKLIST = {"here", "there", "it", "its", "who", "which", "where"}
    if token.text.lower() in BLACKLIST:
        return False
    return True


def get_full_entity(t, doc_ents):
    """
    Return the full entity span text for a given token,
    looking up the entity list from the spaCy doc.
    """
    for ent in doc_ents:
        if ent.start <= t.i < ent.end:
            return ent.text
    return t.text


def get_complete_chunk(token):
    """Return the full noun chunk subtree as a string."""
    return " ".join([t.text for t in token.subtree])


# --------------------- NER & TRIPLET EXTRACTION ----------------------

def name_entity_recognition(text):
    """
    Extract (subject, relation, object) triplets from text using spaCy.
    Only keeps triplets where both subject and object are named entities
    belonging to the TARGET_ENTITIES set.
    Returns a list of dicts with keys:
        subject, subject_type, relation, object, object_type, context
    """
    doc = nlp(text)
    triplets = []
    TARGET_ENTITIES = {"PERSON", "ORG", "GPE", "DATE", "LOC", "EVENT",
                       "WORK_OF_ART", "FAC", "PRODUCT", "NORP"}

    for token in doc:
        if token.pos_ in ["VERB", "AUX"]:
            subj = None
            obj = None
            for child in token.children:
                if child.dep_ in ["nsubj", "nsubjpass"]:
                    subj = child
                elif child.dep_ in ["dobj", "acomp", "attr"]:
                    obj = child
                    for grandchild in child.children:
                        if grandchild.dep_ == "prep":
                            for greatgrandchild in grandchild.children:
                                if greatgrandchild.dep_ == "pobj":
                                    obj = greatgrandchild
                elif child.dep_ == "prep":
                    for grandchild in child.children:
                        if grandchild.dep_ == "pobj":
                            obj = grandchild

            if subj and obj:
                if not (is_clean_token(subj) and is_clean_token(obj)):
                    continue
                if subj.ent_type_ in TARGET_ENTITIES and obj.ent_type_ in TARGET_ENTITIES:
                    final_subj = get_full_entity(subj, doc.ents)
                    final_obj = get_full_entity(obj, doc.ents)
                    NOISE_WORDS = {'the', 'a', 'an', 'and', 'or', 'also', 'then'}
                    relation_lemmas = [token.lemma_.lower()]
                    for t in doc[token.i + 1: obj.i]:
                        if t.ent_type_:
                            continue
                        if t.is_punct or t.is_space or t.is_digit:
                            continue
                        word = t.lemma_.lower()
                        if word not in NOISE_WORDS:
                            clean_word = re.sub(r'[^\w\s]', '', word).strip()
                            if clean_word and clean_word not in NOISE_WORDS:
                                relation_lemmas.append(clean_word)
                    final_relation = " ".join(relation_lemmas).strip()
                    if final_relation:
                        full_context = get_extended_context(token, doc, window=0)
                        triplets.append({
                            "subject": final_subj,
                            "subject_type": subj.ent_type_,
                            "relation": final_relation,
                            "object": final_obj,
                            "object_type": obj.ent_type_,
                            "context": full_context
                        })
    return triplets


# --------------------- RELATION STANDARDIZATION ----------------------

# Royal ontology: standard relation labels for semantic matching
ROYAL_RELATIONS = [
    "BORN_ON_DATE", "BORN_IN_LOCATION", "BORN_AT_PALACE", "CHRISTENED_BY", "CHRISTENED_AT",
    "DIED_ON_DATE", "DIED_IN_LOCATION", "DIED_AT_AGE", "DIED_OF_DISEASE", "BURIED_IN",
    "ASSASSINATION_ATTEMPT_BY", "SUFFERED_FROM", "NURSED_BY", "TUTORED_BY", "EDUCATED_AT",
    "CHILD_OF", "MOTHER_OF", "FATHER_OF", "SIBLING_OF", "SPOUSE_OF",
    "MARRIED_IN_YEAR", "MARRIED_AT", "ENGAGED_TO", "WIDOW_OF",
    "MATERNAL_UNCLE_OF", "FIRST_COUSIN_OF", "NEPHEW_OF", "NIECE_OF", "GRANDCHILD_OF",
    "FATHER_IN_LAW_OF", "MOTHER_IN_LAW_OF", "DESCENDANT_OF", "ANCESTOR_OF",
    "QUEEN_OF", "KING_OF", "PRINCE_OF", "PRINCESS_OF", "DUKE_OF", "DUCHESS_OF",
    "EMPEROR_OF", "EMPRESS_OF", "MARQUESS_OF", "EARL_OF", "BARON_OF",
    "HEIR_PRESUMPTIVE_TO", "HEIR_APPARENT_TO", "THIRD_IN_LINE_TO", "SUCCEEDED_BY",
    "SUCCEEDED_TO_THRONE", "CROWNED_AT", "REIGNED_UNTIL", "GRANTED_TITLE", "ABDICATED_IN",
    "VICEROY_OF", "GOVERNOR_GENERAL_OF", "CONFUCIAN_RULER",
    "PRIME_MINISTER_UNDER", "APPOINTED_BY", "RESIGNED_IN", "VOTED_AGAINST_BILL",
    "ADVISER_TO", "COMPTROLLER_FOR", "LADY_IN_WAITING_TO", "PRIVATE_SECRETARY_TO",
    "FAVOURITE_OF", "RUMOURED_LOVER_OF", "OSTRACIZED_BY", "SUPPORTED_BY",
    "FOUGHT_IN_WAR", "COMMANDED_BY", "PROMOTED_TO_RANK", "ALLIED_WITH",
    "DEPOSED_IN_REVOLUTION", "EXILED_TO", "FLED_TO", "SIGNED_TREATY",
    "MET_WITH_LEADER", "HOSTED_VISIT_OF",
    "RESIDED_AT", "HOLIDAYED_IN", "TRAVELLED_TO", "RETIRED_TO", "OWNER_OF_ESTATE",
    "BOUGHT_PROPERTY", "MOVED_TO", "NAMED_AFTER",
    "GRAND_MASTER_OF_ORDER", "KNIGHT_OF_ORDER", "MEMBER_OF_SOCIETY", "PRESIDENT_OF",
    "FOUNDED_ORGANIZATION", "PATRON_OF", "WROTE_BOOK", "PUBLISHED_JOURNAL",
    "PAINTED_BY", "SCULPTED_BY", "COMMISSIONED_ART", "DEDICATED_TO"
]

# Load the sentence transformer model once at module level
embedder = SentenceTransformer('all-MiniLM-L6-v2')
standard_embeddings = embedder.encode(ROYAL_RELATIONS, convert_to_tensor=True)


def standardize_relation(extracted_relation, threshold=0.55):
    """
    Encode the extracted relation and compute cosine similarity
    against the royal ontology. Returns the best-matching standard
    relation if the score exceeds the threshold, else None.
    """
    if not extracted_relation:
        return None
    query_embedding = embedder.encode(extracted_relation, convert_to_tensor=True)
    cos_scores = util.cos_sim(query_embedding, standard_embeddings)[0]
    top_result = torch.topk(cos_scores, k=1)
    score = top_result[0].item()
    index = top_result[1].item()
    if score >= threshold:
        return ROYAL_RELATIONS[index]
    return None


def extract_knowledge(input_file, output_file, similarity_threshold=0.75):
    """
    Read a JSONL file (crawler output), run NER on each document,
    and write all extracted triplets to a CSV file.
    Each row contains: source_url, subject, subject_type,
    relation, object, object_type, context.
    """
    with open(output_file, mode='w', newline='', encoding='utf-8') as csv_file:
        fieldnames = ['source_url', 'subject', 'subject_type',
                      'relation', 'object', 'object_type', 'context']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        print(f"Starting extraction from: {input_file}")
        count = 0
        rejected_count = 0
        with open(input_file, 'r', encoding='utf-8') as jsonl_file:
            for line in jsonl_file:
                try:
                    data = json.loads(line)
                    url = data.get("url", "unknown")
                    text = data.get("text", "")
                    triples = name_entity_recognition(text)
                    for trip in triples:
                        extracted_rel = trip["relation"]
                        standard_rel = standardize_relation(
                            extracted_rel, threshold=similarity_threshold)
                        if standard_rel or 1 == 1:
                            row = {
                                "source_url": url,
                                "subject": trip["subject"],
                                "subject_type": trip["subject_type"],
                                "relation": trip["relation"],
                                "object": trip["object"],
                                "object_type": trip["object_type"],
                                "context": trip["context"]
                            }
                            writer.writerow(row)
                            count += 1
                        else:
                            rejected_count += 1
                    if (count + rejected_count) % 100 == 0:
                        print(f"Progress: {count} validated, {rejected_count} rejected.")
                except json.JSONDecodeError:
                    print("JSON decode error on a line, skipping.")
                    continue
    print(f"\nOutput saved: {output_file}")
    print(f"Total validated relations: {count}")
    print(f"Total rejected relations: {rejected_count}")
