
# MODULE 5 - RAG over RDF/SPARQL
# Functions:
#   - get_schema_summary(graph)
#   - ask_ollama(prompt, model)
#   - clean_sparql_query(raw_text)
#   - generate_sparql(question, schema_summary, model)
#   - execute_sparql_with_repair(g, question, schema_summary, model, max_attempts)
#   - graph_rag_pipeline(question, g, schema_summary, model)
#   - load_graph(ttl_path)
#   - run_benchmark(ttl_path, questions, model)

import re
import rdflib
import requests

# --- Configuration ---
OLLAMA_URL = "http://localhost:11434/api/generate"
PRIV_NS    = "http://example.org/private#"
MAX_PROPS  = 60   # max properties listed in schema summary


# 1. SCHEMA SUMMARY

def get_schema_summary(graph: rdflib.Graph) -> str:
    """
    Build a compact description of the RDF graph to inject into the LLM prompt.
    - Lists only real prefixes present in the graph (no brick:/csvw: etc.)
    - Lists real priv: properties so the LLM does not invent predicates
    - Includes 30 concrete triples so the LLM sees the exact URI spelling
    """
    # Real prefixes only
    prefixes = "\n".join(
        f"PREFIX {p}: <{n}>"
        for p, n in graph.namespaces()
        if str(n) in (PRIV_NS,
                      "http://www.w3.org/2002/07/owl#",
                      "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
                      "http://www.w3.org/2000/01/rdf-schema#",
                      "http://www.w3.org/2001/XMLSchema#")
    )
    if not prefixes.strip():
        prefixes = (
            f"PREFIX priv: <{PRIV_NS}>\n"
            "PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>\n"
            "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n"
            "PREFIX owl:  <http://www.w3.org/2002/07/owl#>\n"
            "PREFIX xsd:  <http://www.w3.org/2001/XMLSchema#>"
        )

    # Real priv: properties (capped at MAX_PROPS)
    props = sorted(set(
        str(p).split("#")[-1]
        for s, p, o in graph
        if str(p).startswith(PRIV_NS)
    ))[:MAX_PROPS]
    props_block = "\n".join(f"  {p}" for p in props)

    # Classes
    classes = sorted(set(
        str(o).split("#")[-1]
        for s, p, o in graph.triples((None, rdflib.RDF.type, None))
        if str(o).startswith(PRIV_NS)
    ))
    classes_block = "  " + "  ".join(classes)

    # 30 concrete triples - the most useful thing for the LLM:
    # it sees the exact URI spelling of entities and property names
    q_triples = f"""
    SELECT ?s ?p ?o WHERE {{
        ?s a <{PRIV_NS}PERSON> .
        ?s ?p ?o .
        FILTER(STRSTARTS(STR(?p), "{PRIV_NS}"))
        FILTER(STRSTARTS(STR(?o), "{PRIV_NS}"))
    }} LIMIT 30
    """
    triple_lines = []
    try:
        for row in graph.query(q_triples):
            s = str(row.s).split("#")[-1]
            p = str(row.p).split("#")[-1]
            o = str(row.o).split("#")[-1]
            triple_lines.append(f"  priv:{s}  priv:{p}  priv:{o} .")
    except Exception:
        pass
    triples_block = "\n".join(triple_lines) if triple_lines else "  (no examples)"

    return f"""{prefixes}

# Available properties - USE ONLY THESE, never invent names like beFatherOf
{props_block}

# Available classes
{classes_block}

# Concrete sample triples (exact URI and property spelling from the real graph)
{triples_block}

# RULES:
#   - All URIs use the priv: prefix
#   - Entity names use underscores: "George IV" -> priv:George_IV
#   - ALWAYS search entities with FILTER(regex(...)) - never guess URIs directly
#   - NEVER use prefixes not listed here (no brick:, dc:, schema:, csvw:)
"""

# 2. LLM CONNECTION
def ask_ollama(prompt: str, model: str = "llama3:latest") -> str:
    """Send a prompt to the local Ollama API and return the response text."""
    payload = {"model": model, "prompt": prompt, "stream": False}
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        return response.json().get("response", "")
    except Exception as e:
        return f"Ollama connection error: {e}"

# 3. SPARQL CLEANING
def clean_sparql_query(raw_text: str) -> str:
    """
    Extract only the SPARQL block from the LLM response.
    Handles fenced (```sparql.``` or ``````) and unfenced output.
    Cuts everything after the last closing brace to remove post-query text.
    """
    # Case 1: fenced block with any language tag
    match_fence = re.search(r"```(?:[a-zA-Z]*)?\s*(SELECT|PREFIX|ASK|CONSTRUCT|DESCRIBE.*?)```",
                            raw_text, re.IGNORECASE | re.DOTALL)
    if match_fence:
        return match_fence.group(0).replace("```", "").strip()

    # Case 2: fenced block generic
    match_fence2 = re.search(r"```(?:[a-zA-Z]*)?\s*(.*?)```",
                             raw_text, re.IGNORECASE | re.DOTALL)
    if match_fence2:
        candidate = match_fence2.group(1).strip()
        if re.search(r"SELECT|PREFIX|ASK|CONSTRUCT|DESCRIBE", candidate, re.IGNORECASE):
            return candidate

    # Case 3: no backticks - find first SPARQL keyword
    match_kw = re.search(r"(PREFIX|SELECT|ASK|CONSTRUCT|DESCRIBE)", raw_text, re.IGNORECASE)
    if match_kw:
        query = raw_text[match_kw.start():]
        last_brace = query.rfind("}")
        if last_brace != -1:
            query = query[: last_brace + 1]
        return query.strip()

    return raw_text.strip()

# 4. SPARQL GENERATION
def generate_sparql(question: str, schema_summary: str,
                    model: str = "llama3:latest") -> str:
    """
    Translate a natural-language question into a SPARQL 1.1 SELECT query.

    Design choices:
    - Standard prompt with full examples between ```sparql fences
    - Forces FILTER(regex()) to find entities - never hardcoded URIs
    - Provides the real property names used in the graph
    - Does NOT open a fence in the prompt (avoids the "SELECT only" bug
      where the model stops as soon as it sees the opening backticks)
    """
    prompt = f"""You are a SPARQL 1.1 generator for an RDF knowledge graph.

SCHEMA (use ONLY these properties - never invent new ones):
{schema_summary}

PROPERTY CHEATSHEET (real names in this graph):
  father / parent of someone  ->  priv:father  OR  priv:parent
  mother of someone           ->  priv:beMotherOf  OR  priv:parent
  spouse / married to         ->  priv:spouse  OR  priv:marryOf  OR  priv:spouses
  birthplace                  ->  priv:birthPlace
  deathplace                  ->  priv:deathPlace
  successor                   ->  priv:successor
  predecessor                 ->  priv:predecessor
  ancestor                    ->  priv:ancestor

RULES:
1. ALWAYS find the entity with FILTER(regex(str(?entity), "keyword", "i")).
   NEVER write priv:SomeName directly - URI encoding is unpredictable.
2. Use UNION to search in both directions.
3. For spouse/marriage, UNION all three properties (spouse, marryOf, spouses).
4. Output ONLY the SPARQL query inside a ```sparql block. Nothing else.

EXAMPLES:

Q: Who is the father of Alexandra of Denmark?
```sparql
SELECT ?answer WHERE {{
  {{ ?entity priv:father ?answer . }}
  UNION
  {{ ?answer priv:father ?entity . }}
  FILTER(regex(str(?entity), "alexandra.*denmark", "i"))
}}
```

Q: Who is the spouse of George III?
```sparql
SELECT ?answer WHERE {{
  {{ ?entity priv:spouse ?answer . }}
  UNION
  {{ ?answer priv:spouse ?entity . }}
  UNION
  {{ ?entity priv:marryOf ?answer . }}
  UNION
  {{ ?answer priv:marryOf ?entity . }}
  FILTER(regex(str(?entity), "george.*iii", "i"))
}}
```

Q: Where was François Guizot born?
```sparql
SELECT ?answer WHERE {{
  {{ ?entity priv:birthPlace ?answer . }}
  UNION
  {{ ?answer priv:birthPlace ?entity . }}
  FILTER(regex(str(?entity), "guizot", "i"))
}}
```

Q: Is Robert Gascoyne-Cecil married? If so, to whom?
```sparql
SELECT ?answer WHERE {{
  {{ ?entity priv:spouse ?answer . }}
  UNION
  {{ ?answer priv:spouse ?entity . }}
  UNION
  {{ ?entity priv:marryOf ?answer . }}
  UNION
  {{ ?answer priv:marryOf ?entity . }}
  FILTER(regex(str(?entity), "gascoyne", "i"))
}}
```

Q: Who are the parents of Queen Victoria?
```sparql
SELECT ?answer WHERE {{
  {{ ?entity priv:parent ?answer . }}
  UNION
  {{ ?answer priv:parent ?entity . }}
  FILTER(regex(str(?entity), "victoria", "i"))
}}
```

Q: Who is the successor of George III?
```sparql
SELECT ?answer WHERE {{
  {{ ?entity priv:successor ?answer . }}
  UNION
  {{ ?answer priv:successor ?entity . }}
  FILTER(regex(str(?entity), "george.*iii", "i"))
}}
```

Now translate the following question. Output ONLY the ```sparql block:

QUESTION: {question}"""

    raw = ask_ollama(prompt, model)
    return clean_sparql_query(raw)


# 5. SPARQL EXECUTION + SELF-REPAIR

def execute_sparql_with_repair(g: rdflib.Graph, question: str,
                                schema_summary: str,
                                model: str = "llama3:latest",
                                max_attempts: int = 3) -> dict:
    """
    Execute the SPARQL query with automatic self-repair.

    Repair triggers:
    - Python exception (syntax error caught by rdflib)
    - Query executes but returns 0 results (wrong property or regex)

    Returns a dict: query, results, repaired (bool), error, attempts.
    """
    current_query = generate_sparql(question, schema_summary, model)
    repaired = False

    for attempt in range(max_attempts):
        try:
            results = list(g.query(current_query))

            # Valid syntax but empty results -> ask the LLM to fix the property or regex
            if len(results) == 0 and attempt < max_attempts - 1:
                repair_prompt = f"""Your SPARQL query returned ZERO results.

FAILED QUERY:
{current_query}

QUESTION: {question}

MOST LIKELY CAUSES AND FIXES:
1. Wrong property name -> use priv:father, priv:parent, priv:spouse, priv:marryOf,
   priv:spouses, priv:birthPlace, priv:successor, priv:predecessor, priv:ancestor
2. Too specific regex -> simplify it (e.g. "gascoyne" instead of "gascoyne-cecil")
3. Missing direction -> always use UNION for both directions

Write the corrected SPARQL query inside a ```sparql block. Nothing else."""

                current_query = clean_sparql_query(ask_ollama(repair_prompt, model))
                repaired = True
                continue

            return {"query": current_query, "results": results,
                    "repaired": repaired, "error": None, "attempts": attempt + 1}

        except Exception as e:
            if attempt < max_attempts - 1:
                repair_prompt = f"""Your SPARQL has a syntax error.

FAILED QUERY:
{current_query}

ERROR: {e}

QUESTION: {question}

Fix the syntax and write the corrected query inside a ```sparql block. Nothing else."""
                current_query = clean_sparql_query(ask_ollama(repair_prompt, model))
                repaired = True
            else:
                return {"query": current_query, "results": [],
                        "repaired": repaired, "error": str(e),
                        "attempts": attempt + 1}

    return {"query": current_query, "results": [],
            "repaired": repaired,
            "error": "Max attempts reached with 0 results",
            "attempts": max_attempts}


# 6. FULL RAG PIPELINE

def graph_rag_pipeline(question: str, g: rdflib.Graph,
                        schema_summary: str,
                        model: str = "llama3:latest") -> tuple:
    """
    Full pipeline: NL -> SPARQL -> execution -> natural language answer.
    Returns (response_text, final_sparql_query, was_repaired).
    """
    result       = execute_sparql_with_repair(g, question, schema_summary, model)
    final_query  = result["query"]
    was_repaired = result["repaired"]
    results      = result["results"]
    error        = result["error"]

    if results:
        # Check if the LLM hallucinated an ASK query (which returns a boolean)
        if isinstance(results[0], bool):
            facts = f"Boolean Answer from graph: {results[0]}"
        else:
            # Handle standard SELECT query rows
            facts = "\n".join(" | ".join(str(cell) for cell in row) for row in results[:20])
        context = f"Verified facts extracted from the knowledge base:\n{facts}"
    elif error:
        context = f"Query execution error: {error}"
    else:
        context = "No information found in the knowledge base for this question."

    final_prompt = f"""You are a historical assistant.
Use ONLY the context below to answer the question.
If the information is not in the context, say "I don't know, this information is not in the knowledge base."
Do not mention the technical process.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

    response = ask_ollama(final_prompt, model)
    return response, final_query, was_repaired

# 7. GRAPH LOADER
def load_graph(ttl_path: str) -> rdflib.Graph:
    """Load an RDF knowledge graph from a Turtle file into memory."""
    print("Loading graph into memory (done only once)")
    g = rdflib.Graph()
    g.parse(ttl_path, format="turtle")
    print(f"  -> {len(g)} triples loaded from {ttl_path}")
    return g

# 8. BENCHMARK (Baseline vs Graph-RAG)

def run_benchmark(ttl_path: str, questions: list, model: str = "llama3:latest"):
    """
    Run the Baseline vs Graph-RAG benchmark.

    `questions` is a list of strings. Each string is the precise factual
    question asked to the RAG system.

    The baseline receives a DIFFERENT, broader version of the same question
    (written directly in the list as a second element of a tuple, or auto-
    generated if only a string is provided). This makes the comparison fair:
    the baseline answers from general knowledge on a vague question, while the
    RAG answers the precise question using the knowledge graph.

    Two usage modes:
      # Mode 1 - plain strings (baseline question auto-generated)
      questions = ["Who is the parent of Alexandra of Denmark?", ]

      # Mode 2 - explicit tuples (baseline_question, rag_question)
      questions = [
          ("Tell me about the Danish royal family.",
           "Who is the parent of Alexandra of Denmark?"),
          
      ]
    """
    g      = load_graph(ttl_path)
    schema = get_schema_summary(g)

    print(f"{'BENCHMARK: BASELINE VS GRAPH-RAG':}")

    for i, entry in enumerate(questions, 1):

        # --- Unpack question pair ---
        if isinstance(entry, tuple) and len(entry) == 2:
            baseline_q, rag_q = entry
        else:
            rag_q      = str(entry)
            baseline_q = rag_q   # same question, no KB - shows hallucination

        print(f"\nQUESTION {i}: {rag_q}\n")

        # --- Baseline: LLM only, no knowledge base ---
        baseline_answer = ask_ollama(
            f"Answer this question as best as you can: {baseline_q}", model)
        print(f"BASELINE: {baseline_answer.strip()}\n")

        # --- Graph-RAG: NL -> SPARQL -> KB -> LLM ---
        response, final_query, was_repaired = graph_rag_pipeline(
            rag_q, g, schema, model)
        print(f"SPARQL generated:\n{final_query}\n")
        print(f"Self-repair triggered? {'Yes' if was_repaired else 'No'}")
        print(f"RAG ANSWER:\n{response.strip()}")
