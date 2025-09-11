import random

def rule_based(product_name):
    """Rule-based system: returns candidate segments."""
    print("[Rule-based] Processing:", product_name)
    return ["segmentA", "segmentB"]

def classifier(product_name):
    """Classifier model: predicts seg/fam/cls."""
    print("[Classifier] Processing:", product_name)
    return {
        "segment": "segmentC",
        "family": "familyX",
        "class": "class1"
    }

def embedding(input_data, source="classifier"):
    """Embedding model: maps to seg/fam/cls."""
    print(f"[Embedding-{source}] Processing:", input_data)
    return {
        "segment": random.choice(["segmentA", "segmentB", "segmentC"]),
        "family": random.choice(["familyX", "familyY"]),
        "class": random.choice(["class1", "class2"])
    }

def database_lookup():
    """Database of all seg/fam/cls."""
    print("[Database] Fetching all candidates...")
    return {
        "segments": ["segmentA", "segmentB", "segmentC"],
        "families": ["familyX", "familyY"],
        "classes": ["class1", "class2"]
    }

def voting(predictions):
    """Voting model: combines predictions."""
    print("[Voting] Aggregating predictions...")
    # Just choose majority (dummy logic)
    result = {
        "segment": random.choice([p["segment"] for p in predictions]),
        "family": random.choice([p["family"] for p in predictions]),
        "class": random.choice([p["class"] for p in predictions]),
    }
    return result


def pipeline(product_name):
    print("\n--- Pipeline Start ---")

    # Step 1: Rule-based
    rule_segments = rule_based(product_name)
    rb_emb = embedding(rule_segments, source="rule")

    # Step 2: Classifier
    clf_pred = classifier(product_name)
    clf_emb = embedding(clf_pred, source="classifier")

    # Step 3: Database + Embedding
    db_candidates = database_lookup()
    db_emb = embedding(db_candidates, source="database")

    # Step 4: Voting
    final_output = voting([rb_emb, clf_emb, db_emb])

    print("--- Pipeline End ---")

    return final_output


# ----------------------------
# Run Example
# ----------------------------
if __name__ == "__main__":
    product_name = "Laptop Charger"
    output = pipeline(product_name)
    print("\nFinal Output:", output)
