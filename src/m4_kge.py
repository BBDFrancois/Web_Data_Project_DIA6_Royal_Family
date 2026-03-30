
# MODULE 4 - Knowledge Graph Embeddings (KGE)
# Functions:
#   - load_triples_from_txt(path)
#   - subsample_triples(triples, n, seed)
#   - triples_to_numpy(triples)
#   - prepare_datasets(kge_data_folder)
#   - train_model(model_name, tf_train, tf_valid, tf_test, )
#   - print_comparison_table(results)
#   - run_size_sensitivity(train_raw, best_model_name, )
#   - get_entity_embeddings(model_result)
#   - plot_tsne(embeddings, entity_to_id, class_map, title)
#   - nearest_neighbors(embeddings, entity_to_id, query_entities, k)
#   - analyze_relation_behavior(model_result, sample_n)
#   - swrl_vs_embedding_comparison(model_result)
#   - run_full_kge_pipeline(kge_data_folder, )

import os
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory

from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

import torch


# DATA UTILITIES

def load_triples_from_txt(path: str) -> list:
    """
    Read a KGE dataset file (TSV format: head TAB relation TAB tail).
    Returns a list of (head, relation, tail) string tuples.
    """
    triples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) == 3:
                triples.append(tuple(parts))
    return triples


def subsample_triples(triples: list, n: int, seed: int = 42) -> list:
    """Randomly subsample n triples from a list (for size-sensitivity analysis)."""
    rng = random.Random(seed)
    if n >= len(triples):
        return triples
    return rng.sample(triples, n)


def triples_to_numpy(triples: list) -> np.ndarray:
    """Convert a list of (h, r, t) string tuples to a numpy array."""
    return np.array(triples, dtype=str)


# DATASET PREPARATION

def prepare_datasets(kge_data_folder: str):
    """
    Load train/valid/test.txt and build PyKEEN TriplesFactory objects.
    Removes entities from valid/test that do not appear in train
    to guarantee no entity leakage between splits.
    Returns: tf_train, tf_valid, tf_test, train_raw (list of tuples)
    """
    print("STEP 1 - Dataset preparation")

    train_path = os.path.join(kge_data_folder, "train.txt")
    valid_path = os.path.join(kge_data_folder, "valid.txt")
    test_path  = os.path.join(kge_data_folder, "test.txt")

    train_raw = load_triples_from_txt(train_path)
    valid_raw = load_triples_from_txt(valid_path)
    test_raw  = load_triples_from_txt(test_path)

    print(f"  Train : {len(train_raw):>7} triples")
    print(f"  Valid : {len(valid_raw):>7} triples")
    print(f"  Test  : {len(test_raw):>7} triples")

    # Remove entities present only in valid/test (no leakage guarantee)
    train_entities = (set(h for h, r, t in train_raw) |
                      set(t for h, r, t in train_raw))
    valid_raw = [(h, r, t) for h, r, t in valid_raw
                 if h in train_entities and t in train_entities]
    test_raw  = [(h, r, t) for h, r, t in test_raw
                 if h in train_entities and t in train_entities]
    print(f"  After isolated entity removal - Valid: {len(valid_raw)}, Test: {len(test_raw)}")

    print(f"DEBUG SPLITS - Train: {len(train_raw)}, Valid: {len(valid_raw)}, Test: {len(test_raw)}")

    tf_train = TriplesFactory.from_labeled_triples(triples_to_numpy(train_raw))
    tf_valid = TriplesFactory.from_labeled_triples(
        triples_to_numpy(valid_raw),
        entity_to_id=tf_train.entity_to_id,
        relation_to_id=tf_train.relation_to_id,
    )
    tf_test = TriplesFactory.from_labeled_triples(
        triples_to_numpy(test_raw),
        entity_to_id=tf_train.entity_to_id,
        relation_to_id=tf_train.relation_to_id,
    )

    print(f"\n  Unique entities  : {tf_train.num_entities}")
    print(f"  Unique relations : {tf_train.num_relations}")

    return tf_train, tf_valid, tf_test, train_raw


# MODEL TRAINING

def train_model(model_name: str, tf_train, tf_valid, tf_test,
                embedding_dim: int = 100, epochs: int = 200,
                lr: float = 0.01, batch_size: int = 512) -> dict:
    """
    Train a KGE model (TransE or DistMult) using PyKEEN pipeline.
    Configuration is kept identical across models for fair comparison.
    Returns a dict with: model_name, result, metrics (MRR, Hits@1/3/10).
    """
    print(f"  Training: {model_name}  (dim={embedding_dim}, epochs={epochs})")

    result = pipeline(
        training=tf_train,
        validation=tf_valid,
        testing=tf_test,
        model=model_name,
        model_kwargs={"embedding_dim": embedding_dim},
        optimizer="Adam",
        optimizer_kwargs={"lr": lr},
        training_kwargs={"num_epochs": epochs, "batch_size": batch_size},
        negative_sampler="basic",
        evaluator_kwargs={"filtered": True},
        random_seed=42,
    )

    # --- Metric extraction ---
    # PyKEEN changed its metric API across versions.
    # Strategy: try the direct attribute access first (most reliable),
    # then fall back to to_dict() key scanning.
    def _extract_metrics(result) -> dict:
        mr = result.metric_results

        # Approach 1 - direct attribute access (works in PyKEEN >= 1.8)
        try:
            return {
                "MRR":    float(mr.get_metric("inverse_harmonic_mean_rank")),
                "Hits@1": float(mr.get_metric("hits_at_1")),
                "Hits@3": float(mr.get_metric("hits_at_3")),
                "Hits@10":float(mr.get_metric("hits_at_10")),
            }
        except Exception:
            pass

        # Approach 2 - iterate the RankBasedMetricResults dataframe
        try:
            df = mr.to_df()
            def _from_df(df, metric):
                rows = df[df["Metric"].str.lower().str.contains(metric, na=False)]
                if not rows.empty:
                    return float(rows["Value"].iloc[0])
                return float("nan")
            return {
                "MRR":    _from_df(df, "inverse_harmonic_mean_rank"),
                "Hits@1": _from_df(df, "hits_at_1"),
                "Hits@3": _from_df(df, "hits_at_3"),
                "Hits@10":_from_df(df, "hits_at_10"),
            }
        except Exception:
            pass

        # Approach 3 - to_dict() key scan (last resort)
        try:
            d = mr.to_dict()
            def _find(d, *suffixes):
                for suffix in suffixes:
                    for k, v in d.items():
                        if k.lower().endswith(suffix.lower()) and v is not None:
                            try:
                                return float(v)
                            except (TypeError, ValueError):
                                continue
                return float("nan")
            return {
                "MRR":    _find(d, "inverse_harmonic_mean_rank", "mean_reciprocal_rank"),
                "Hits@1": _find(d, "hits_at_1", "hits@1"),
                "Hits@3": _find(d, "hits_at_3", "hits@3"),
                "Hits@10":_find(d, "hits_at_10", "hits@10"),
            }
        except Exception:
            pass

        # Approach 4 - inspect all numeric attributes of the result object
        print(f"  [DEBUG] All approaches failed. Inspecting metric_results attributes:")
        for attr in dir(mr):
            if not attr.startswith("_"):
                try:
                    val = getattr(mr, attr)
                    if isinstance(val, float):
                        print(f"    {attr} = {val}")
                except Exception:
                    pass
        return {"MRR": float("nan"), "Hits@1": float("nan"),
                "Hits@3": float("nan"), "Hits@10": float("nan")}

    df_metrics = result.metric_results.to_df()
    print("\n--- DEBUG METRICS DATAFRAME ICIIIII---")
    print(df_metrics[['Metric', 'Value']].head(20))
    print("-------------------------------\n")

    metrics_dict = _extract_metrics(result)
    mrr    = metrics_dict["MRR"]
    hits1  = metrics_dict["Hits@1"]
    hits3  = metrics_dict["Hits@3"]
    hits10 = metrics_dict["Hits@10"]

    print(f"\n  ── Results {model_name} ──")
    print(f"  MRR    : {mrr:.4f}")
    print(f"  Hits@1 : {hits1:.4f}")
    print(f"  Hits@3 : {hits3:.4f}")
    print(f"  Hits@10: {hits10:.4f}")

    return {
        "model_name": model_name,
        "result": result,
        "metrics": {"MRR": mrr, "Hits@1": hits1, "Hits@3": hits3, "Hits@10": hits10},
        "embedding_dim": embedding_dim,
    }

# COMPARISON TABLE

def print_comparison_table(results: list):
    """Print a formatted comparison table of KGE model metrics."""
    print("COMPARISON TABLE - TransE vs DistMult")
    header = f"{'Model':<12} | {'MRR':>8} | {'Hits@1':>8} | {'Hits@3':>8} | {'Hits@10':>8}"
    print(header)
    for r in results:
        m = r["metrics"]
        print(f"{r['model_name']:<12} | "
              f"{m['MRR']:>8.4f} | "
              f"{m['Hits@1']:>8.4f} | "
              f"{m['Hits@3']:>8.4f} | "
              f"{m['Hits@10']:>8.4f}")
    print()


# SIZE SENSITIVITY

def run_size_sensitivity(train_raw: list, best_model_name: str = "TransE",
                          embedding_dim: int = 100, epochs: int = 100):
    """
    Retrain the best model on 3 subsets of the training data:
      - 20k triples
      - 50k triples
      - Full dataset
    Plots the final training loss to show how performance scales with KB size.
    """
    print("STEP 5.2 - Size Sensitivity (20k / 50k / full)")

    total = len(train_raw)
    sizes = {
        "20k" : min(20_000, total),
        "50k" : min(50_000, total),
        "full": total,
    }

    sensitivity_results = []

    for label, n in sizes.items():
        # Build a minimal train/test split from the subset
        # (PyKEEN requires testing= even for loss-only tracking)
        subset    = subsample_triples(train_raw, n)
        random.shuffle(subset)
        split_idx = int(len(subset) * 0.9)
        sub_train = subset[:split_idx]
        sub_test  = subset[split_idx:]

        tf_sub   = TriplesFactory.from_labeled_triples(triples_to_numpy(sub_train))
        # Filter test entities to those present in train
        train_ents = set(tf_sub.entity_to_id.keys())
        sub_test   = [(h, r, t) for h, r, t in sub_test
                      if h in train_ents and t in train_ents]
        if not sub_test:
            sub_test = sub_train[:max(1, len(sub_train) // 10)]

        tf_sub_test = TriplesFactory.from_labeled_triples(
            triples_to_numpy(sub_test),
            entity_to_id=tf_sub.entity_to_id,
            relation_to_id=tf_sub.relation_to_id,
        )

        print(f"\n  [{label}] {len(sub_train)} training triples")
        try:
            r = pipeline(
                training=tf_sub,
                testing=tf_sub_test,
                model=best_model_name,
                model_kwargs={"embedding_dim": embedding_dim},
                optimizer_kwargs={"lr": 0.01},
                training_kwargs={"num_epochs": epochs, "batch_size": 512},
                random_seed=42,
            )
            # Extract final training loss - handle all PyKEEN versions
            losses_raw = getattr(r, "losses", None)
            if isinstance(losses_raw, list) and losses_raw:
                loss = float(losses_raw[-1])
            elif isinstance(losses_raw, dict) and losses_raw:
                loss = float(list(losses_raw.values())[-1])
            elif hasattr(r, "loss_per_epoch") and r.loss_per_epoch:
                loss = float(list(r.loss_per_epoch.values())[-1])
            else:
                # Last resort: read from training loop's internal tracker
                tracker = getattr(r.training, "losses", None)
                loss = float(tracker[-1]) if tracker else float("nan")
            print(f"    Final loss: {loss:.4f}")
            sensitivity_results.append({"size": label, "n": len(sub_train), "loss": loss})
        except Exception as e:
            print(f"    [ERROR] {e}")
            sensitivity_results.append({"size": label, "n": n, "loss": float("nan")})

    # Plot - skip entries where loss is nan (subset too small or training failed)
    valid = [(r["size"], r["loss"]) for r in sensitivity_results
             if r["loss"] == r["loss"]]  # nan != nan

    if not valid:
        print("\n  [WARNING] All loss values are NaN - size sensitivity plot skipped.")
        print("  This usually means r.losses is empty in your PyKEEN version.")
        print("  The training still ran correctly; only the loss retrieval failed.")
        return sensitivity_results

    v_labels = [v[0] for v in valid]
    v_losses = [v[1] for v in valid]
    colors   = ["#4C72B0", "#DD8452", "#55A868"][:len(valid)]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(v_labels, v_losses, color=colors)
    ax.set_title(f"Size Sensitivity - {best_model_name}: Final Training Loss")
    ax.set_xlabel("Training dataset size")
    ax.set_ylabel("Loss (last epoch)")
    # Only add text labels when bars have non-zero height
    y_max = max(v_losses) if v_losses else 1
    ax.set_ylim(0, y_max * 1.2 if y_max > 0 else 1)
    for bar, v in zip(bars, v_losses):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + y_max * 0.02,
                f"{v:.3f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig("size_sensitivity.png", dpi=150)
    plt.show()
    print("\n  Plot saved: size_sensitivity.png")
    return sensitivity_results


# EMBEDDING ANALYSIS

def get_entity_embeddings(model_result: dict):
    """
    Extract entity embedding matrix and entity-to-id mapping from a trained model.
    Returns: (embeddings np.ndarray, entity_to_id dict)
    """
    model = model_result["result"].model
    entity_repr = model.entity_representations[0]
    embeddings = entity_repr(indices=None).detach().cpu().numpy()
    entity_to_id = model_result["result"].training.entity_to_id
    return embeddings, entity_to_id


def plot_tsne(embeddings: np.ndarray, entity_to_id: dict,
              class_map: dict = None, title: str = "t-SNE"):
    """
    Reduce entity embeddings to 2D with t-SNE and plot a scatter chart.
    class_map: optional dict {entity_name: class_label} to colour by ontology class.
    Saves the plot as a PNG file.
    """
    print(f"\n  Computing t-SNE on {embeddings.shape[0]} entities")
    n = min(embeddings.shape[0], 3000)
    idx = np.random.choice(embeddings.shape[0], n, replace=False)
    emb_subset = embeddings[idx]

    perplexity = min(30, max(5, n // 4))  # perplexity must be >= 5
    # n_iter was renamed to max_iter in scikit-learn 1.2 - try both for compatibility
    try:
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
    except TypeError:
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=1000)
    coords = tsne.fit_transform(emb_subset)

    id_to_entity = {v: k for k, v in entity_to_id.items()}
    entity_names = [id_to_entity[i] for i in idx]

    fig, ax = plt.subplots(figsize=(12, 8))

    if class_map:
        classes = sorted(set(class_map.values()))
        palette = plt.cm.get_cmap("tab10", len(classes))
        class_to_color = {c: palette(i) for i, c in enumerate(classes)}
        colors = [class_to_color.get(class_map.get(e, "other"), "lightgray")
                  for e in entity_names]
        ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=8, alpha=0.6)
        patches = [mpatches.Patch(color=class_to_color[c], label=c) for c in classes]
        ax.legend(handles=patches, loc="best", fontsize=7)
    else:
        ax.scatter(coords[:, 0], coords[:, 1], s=8, alpha=0.5, color="#4C72B0")

    ax.set_title(title)
    ax.set_xlabel("t-SNE dimension 1")
    ax.set_ylabel("t-SNE dimension 2")
    plt.tight_layout()
    fname = title.replace(" ", "_").lower() + ".png"
    plt.savefig(fname, dpi=150)
    plt.show()
    print(f"  Plot saved: {fname}")


def nearest_neighbors(embeddings: np.ndarray, entity_to_id: dict,
                      query_entities: list, k: int = 5):
    """
    Print the k nearest neighbours (by cosine similarity) for each query entity.
    Useful for qualitative evaluation of semantic coherence in the embedding space.
    """
    id_to_entity = {v: k for k, v in entity_to_id.items()}
    print(f"STEP 6.1 - Nearest Neighbors (k={k})")

    sim_matrix = cosine_similarity(embeddings)

    for qe in query_entities:
        if qe not in entity_to_id:
            print(f"  Entity '{qe}' not found in the model.")
            continue
        qid = entity_to_id[qe]
        sims = sim_matrix[qid]
        top_ids = np.argsort(sims)[::-1][1: k + 1]
        print(f"\n  Neighbors of '{qe}':")
        for rank, nid in enumerate(top_ids, 1):
            print(f"    {rank}. {id_to_entity[nid]:<40} (cosine = {sims[nid]:.4f})")


def analyze_relation_behavior(model_result: dict, sample_n: int = 10):
    """
    Qualitative analysis of relation embeddings:
    - Find the top-N most similar relation pairs (potential inverses or synonyms)
    - Find which relations are most symmetric (vector ≈ its own negation)
    """
    print("STEP 6.3 - Relation Behavior Analysis")

    model = model_result["result"].model
    rel_repr = model.relation_representations[0]
    rel_embs = rel_repr(indices=None).detach().cpu().numpy()
    rel_to_id = model_result["result"].training.relation_to_id

    sim_matrix = cosine_similarity(rel_embs)
    id_to_rel = {v: k for k, v in rel_to_id.items()}
    n_rels = rel_embs.shape[0]

    pairs = []
    for i in range(n_rels):
        for j in range(i + 1, n_rels):
            pairs.append((sim_matrix[i, j], id_to_rel[i], id_to_rel[j]))
    pairs.sort(reverse=True)

    print(f"\n  Top {sample_n} most similar relation pairs (potential inverses/synonyms):")
    for sim, r1, r2 in pairs[:sample_n]:
        print(f"    {r1:<35} <-> {r2:<35} | sim = {sim:.4f}")

    print(f"\n  Potentially symmetric relations (vector ≈ its negation):")
    for i in range(min(n_rels, sample_n)):
        neg_vec = -rel_embs[i]
        sims = cosine_similarity([neg_vec], rel_embs)[0]
        most_similar_id = np.argmax(sims)
        print(f"    -{id_to_rel[i]:<30} ≈ {id_to_rel[most_similar_id]:<30} | sim = {sims[most_similar_id]:.4f}")

# PART 8 - SWRL vs EMBEDDING COMPARISON

def swrl_vs_embedding_comparison(model_result: dict):
    """
    Exercise 8 of the lab: compare SWRL rules with embedding arithmetic.

    Rule 1: parent(?x,?y) ∧ parent(?y,?z)  ancestor(?x,?z)
    Check: vector(parent) + vector(parent) ≈ vector(ancestor)?

    Rule 2: successor(?a,?b)  predecessor(?b,?a)
    Check: vector(successor) ≈ -vector(predecessor)?
    """
    print("STEP 8 - Rule-based vs Embedding Comparison")

    model = model_result["result"].model
    rel_repr = model.relation_representations[0]
    rel_embs = rel_repr(indices=None).detach().cpu().numpy()
    rel_to_id = model_result["result"].training.relation_to_id

    candidates = {
        "parent":      ["parent", "isParentOf", "beMotherOf", "father"],
        "ancestor":    ["ancestor", "isAncestorOf"],
        "successor":   ["successor", "succeedImmediatelyAs"],
        "predecessor": ["predecessor"],
        "spouse":      ["spouses", "spouse", "marryOf"],
    }
    found = {}
    for key, names in candidates.items():
        for name in names:
            if name in rel_to_id:
                found[key] = name
                break

    print(f"\n  Relations found in model: {found}")

    if "parent" in found and "ancestor" in found:
        v_parent   = rel_embs[rel_to_id[found["parent"]]]
        v_ancestor = rel_embs[rel_to_id[found["ancestor"]]]
        v_composed = v_parent + v_parent
        sim = cosine_similarity([v_composed], [v_ancestor])[0][0]
        print(f"\n  SWRL rule: parent(?x,?y) ∧ parent(?y,?z)  ancestor(?x,?z)")
        print(f"  Embedding: vector({found['parent']}) + vector({found['parent']})")
        print(f"           ≈ vector({found['ancestor']}) ?")
        print(f"  Cosine similarity = {sim:.4f}")
        print(f"  {'✓ YES' if sim > 0.5 else '✗ NO'} - embedding "
              f"{'captures' if sim > 0.5 else 'does not capture'} this composition rule.")

    if "successor" in found and "predecessor" in found:
        v_succ = rel_embs[rel_to_id[found["successor"]]]
        v_pred = rel_embs[rel_to_id[found["predecessor"]]]
        sim_inv = cosine_similarity([v_succ], [v_pred])[0][0]
        print(f"\n  SWRL rule: successor(?a,?b)  predecessor(?b,?a)")
        print(f"  Embedding: vector({found['successor']}) ≈ -vector({found['predecessor']}) ?")
        print(f"  Cosine similarity(successor, predecessor) = {sim_inv:.4f}")
        print(f"  (For TransE, expect sim ≈ -1 if the relations are true inverses)")


# MAIN PIPELINE

def run_full_kge_pipeline(kge_data_folder: str = "kge_data",
                           embedding_dim: int = 100, epochs: int = 200,
                           lr: float = 0.01, batch_size: int = 512,
                           query_entities: list = None) -> dict:
    """
    Full KGE pipeline:
      1. Load and clean train/valid/test datasets
      2. Train TransE and DistMult (identical config)
      3. Print comparison table
      4. Size sensitivity analysis (20k / 50k / full)
      5. t-SNE visualisation coloured by ontology class
      6. Nearest neighbors for selected entities
      7. Relation behavior analysis
      8. SWRL vs embedding comparison
    Returns a dict with keys: transe, distmult, best.
    """
    print(" KGE FULL PIPELINE - British Royal Family (19th)")

    # 1. Data
    tf_train, tf_valid, tf_test, train_raw = prepare_datasets(kge_data_folder)

    # 2. Training
    all_results = []
    transe_result = train_model("TransE", tf_train, tf_valid, tf_test,
                                embedding_dim=embedding_dim, epochs=epochs,
                                lr=lr, batch_size=batch_size)
    all_results.append(transe_result)

    distmult_result = train_model("DistMult", tf_train, tf_valid, tf_test,
                                  embedding_dim=embedding_dim, epochs=epochs,
                                  lr=lr, batch_size=batch_size)
    all_results.append(distmult_result)

    # 3. Comparison
    print_comparison_table(all_results)
    best = max(all_results, key=lambda r: r["metrics"]["MRR"])
    print(f"  Best model: {best['model_name']} (MRR = {best['metrics']['MRR']:.4f})")

    # 4. Size sensitivity
    run_size_sensitivity(train_raw, best_model_name=best["model_name"],
                         embedding_dim=embedding_dim, epochs=100)

    # 5. t-SNE
    embeddings, entity_to_id = get_entity_embeddings(best)

    # Print a sample so the user knows the exact entity name format to pass
    # as query_entities (names come from the KGE files, e.g. "QueenVictoria",
    # "George_IV", "WilhelmIi" - they mirror the priv: local names in the TTL)
    print("\n  Sample entity names in the model (use these verbatim for query_entities):")
    for name in list(entity_to_id.keys())[:20]:
        print(f"    \"{name}\"")

    PERSON_KW = {"Queen", "King", "Prince", "Princess", "Duke", "Duchess",
                 "Victoria", "Edward", "Albert", "George", "William", "Alice"}
    GPE_KW = {"Britain", "England", "France", "Germany", "Kingdom", "Empire",
              "Prussia", "Scotland", "Ireland"}
    class_map = {}
    for entity in entity_to_id:
        if any(kw in entity for kw in PERSON_KW):
            class_map[entity] = "PERSON"
        elif any(kw in entity for kw in GPE_KW):
            class_map[entity] = "GPE/LOC"
        else:
            class_map[entity] = "OTHER"

    plot_tsne(embeddings, entity_to_id, class_map=class_map,
              title=f"t-SNE - {best['model_name']} embeddings (coloured by class)")

    # 6. Nearest neighbors
    if query_entities is None:
        query_entities = ["QueenVictoria", "AlbertEdward", "George_IV",
                          "WilhelmIi", "PrinceAlbert", "Edward", "Victoria"]
    # Keep only names that actually exist in the model (exact match)
    matched = [e for e in query_entities if e in entity_to_id]
    if not matched:
        print("\n    None of the requested query_entities were found in the model.")
        print("      Falling back to the first 5 entities in the model.")
        print("      Re-run with the names printed above for targeted results.")
        matched = list(entity_to_id.keys())[:5]
    nearest_neighbors(embeddings, entity_to_id, matched, k=5)

    # 7. Relation behavior
    analyze_relation_behavior(best, sample_n=10)

    # 8. SWRL vs embedding
    swrl_vs_embedding_comparison(best)

    print(" KGE PIPELINE COMPLETE")

    return {"transe": transe_result, "distmult": distmult_result, "best": best}
