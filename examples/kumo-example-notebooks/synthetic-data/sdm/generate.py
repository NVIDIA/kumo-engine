"""Seeded local datasets for SDM examples; no downloads or warehouse writes."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Retain the feature/label contracts of the examples, with small row counts.
SPECS = {
    "bioresponse": (None, "Activity", 2, 128),
    "icr_identify_age_related_conditions": ("Id", "Class", 2, 55),
    "two_sigma_connect_rental_listing_inquiries": (
        "listing_id",
        "interest_level",
        3,
        8,
    ),
    "restaurant_revenue_prediction": ("Id", "revenue", 0, 37),
    "leaf_classification": ("id", "species", 99, 192),
    "forest_cover_type_prediction": ("Id", "Cover_Type", 7, 54),
    "afsis_soil_properties": ("PIDN", "Ca", 0, 512),
    "unimelb_grant_application": ("Grant.Application.ID", "Grant.Status", 2, 64),
    "playground_series_s4e10": ("id", "loan_status", 2, 12),
    "santander_customer_transaction_prediction": ("ID_code", "target", 2, 200),
    "santander_value_prediction_challenge": ("ID", "target", 0, 512),
    "mercari_price_suggestion_challenge": ("train_id", "price", 0, 8),
    "otto_group_product_classification_challenge": ("id", "target", 9, 93),
    "itk_fall_2018": (None, "class", 2, 170),
    "breast_cancer_classification": (None, "target", 2, 30),
    "classification": (None, "target", 10, 64),
    "regression": (None, "MedHouseVal", 0, 8),
    "nearest_neighbor": (None, "class", 2, 20),
}


def tabular(name, seed=42):
    """Create task-specific feature signal with independent label noise."""
    identifier, target, classes, width = SPECS[name]
    n = max(400, int(115000 / width))
    if name == "leaf_classification":
        n = 990
    if name == "two_sigma_connect_rental_listing_inquiries":
        n = 2800
    if name == "mercari_price_suggestion_challenge":
        n = 6000
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, width))
    score = 1.7 * x[:, 0] - x[:, 1] + 0.7 * x[:, 2] ** 2
    noisy_score = score + rng.normal(0, 0.6, n)
    labels = (
        np.minimum(
            (pd.Series(noisy_score).rank(pct=True).to_numpy() * classes).astype(int),
            classes - 1,
        )
        if classes
        else np.exp(4 + 0.3 * noisy_score)
    )
    prefix = {
        "bioresponse": "D",
        "restaurant_revenue_prediction": "P",
        "santander_customer_transaction_prediction": "var_",
        "otto_group_product_classification_challenge": "feat_",
    }.get(name, "feature_")
    frame = pd.DataFrame(x.round(4), columns=[f"{prefix}{i + 1}" for i in range(width)])
    if identifier:
        frame.insert(0, identifier, np.arange(n))
    if name == "leaf_classification":
        frame.columns = ["id"] + [
            f"{kind}{i}"
            for kind in ("margin", "shape", "texture")
            for i in range(1, 65)
        ]
        labels = np.array([f"species_{i:02d}" for i in labels])
    if name == "otto_group_product_classification_challenge":
        labels = np.array([f"Class_{i + 1}" for i in labels])
    if name == "forest_cover_type_prediction":
        labels += 1
    if name == "itk_fall_2018":
        frame = frame.mask(rng.random(frame.shape) < 0.08)
        labels = noisy_score > np.quantile(noisy_score, 0.9)
    if name == "icr_identify_age_related_conditions":
        frame["EJ"] = np.where(x[:, 0] > 0, "A", "B")
    if name == "restaurant_revenue_prediction":
        frame["Open Date"] = (
            pd.Timestamp("2005-01-01")
            + pd.to_timedelta(rng.integers(0, 3000, n), unit="D")
        ).strftime("%m/%d/%Y")
        frame["City"] = rng.choice(["city_a", "city_b", "city_c"], n)
        frame["City Group"] = rng.choice(["large", "small"], n)
        frame["Type"] = rng.choice(["FC", "IL"], n)
    if name == "unimelb_grant_application":
        frame["Start.date"] = (
            pd.Timestamp("2020-01-01")
            + pd.to_timedelta(rng.integers(0, 1000, n), unit="D")
        ).strftime("%d/%m/%y")
    if name == "afsis_soil_properties":
        frame["Depth"] = rng.choice(["topsoil", "subsoil"], n)
        for i, col in enumerate(["P", "pH", "SOC", "Sand"]):
            frame[col] = (x[:, i] + 0.5 * x[:, i + 1] + rng.normal(0, 0.2, n)).round(4)
    if name == "two_sigma_connect_rental_listing_inquiries":
        labels = np.array(["low", "medium", "high"])[labels]
        frame["bedrooms"] = rng.integers(1, 5, n)
        frame["bathrooms"] = rng.integers(1, 3, n)
        frame["price"] = (2000 + 250 * x[:, 0]).round(2)
        frame["description"] = np.where(
            x[:, 0] > 0,
            "Spacious bright rental near transit",
            "Compact quiet rental with courtyard",
        )
        frame["photos"] = [[] for _ in range(n)]
        frame["display_address"] = "synthetic address"
        frame["street_address"] = "synthetic street"
        frame["features"] = [
            ["Dishwasher", "Balcony"] if v > 0 else ["Courtyard"] for v in x[:, 0]
        ]
        frame["created"] = pd.date_range("2025-01-01", periods=n, freq="h").astype(str)
        frame["building_id"] = rng.integers(0, 30, n).astype(str)
        frame["manager_id"] = rng.integers(0, 10, n).astype(str)
    if name == "mercari_price_suggestion_challenge":
        frame["name"] = np.where(
            x[:, 0] > 0, "Premium synthetic item", "Basic synthetic item"
        )
        frame["item_description"] = np.where(
            x[:, 0] > 0,
            "Unused durable item with accessories",
            "Used item with minor wear",
        )
        frame["category_name"] = rng.choice(["Home/Decor", "Sports/Outdoor"], n)
        frame["brand_name"] = rng.choice(["synthetic_a", "synthetic_b"], n)
        frame["item_condition_id"] = rng.integers(1, 5, n)
        frame["shipping"] = rng.integers(0, 2, n)
    frame[target] = labels
    return frame


def generate(name, output, seed=42):
    """Write labeled training and label-free scoring files to a local directory."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    frame = tabular(name, seed)
    target = SPECS[name][1]
    targets = [target] + (
        ["P", "pH", "SOC", "Sand"] if name == "afsis_soil_properties" else []
    )
    # Stratification preserves all 99 species in both populations.
    if SPECS[name][2]:
        test_index = (
            frame.groupby(target, group_keys=False)
            .sample(frac=0.2, random_state=seed)
            .index
        )
    else:
        test_index = frame.sample(frac=0.2, random_state=seed).index
    train = frame.drop(index=test_index)
    test = frame.loc[test_index].drop(columns=targets)
    if name == "two_sigma_connect_rental_listing_inquiries":
        train.to_json(output / "train.json")
        test.to_json(output / "test.json")
    elif name == "itk_fall_2018":
        for part, filename in [
            (train, "aps_failure_training_set.arff"),
            (test.assign(**{"class": False}), "aps_failure_test_set.arff"),
        ]:
            header = (
                "@RELATION synthetic\n"
                + "\n".join(
                    f"@ATTRIBUTE {c} {'{false,true}' if c == 'class' else 'NUMERIC'}"
                    for c in part
                )
                + "\n@DATA\n"
            )
            part = part.copy()
            part["class"] = part["class"].map({False: "false", True: "true"})
            (output / filename).write_text(
                header + part.to_csv(index=False, header=False, na_rep="?")
            )
    else:
        train_name, test_name, sep = "train.csv", "test.csv", ","
        if name == "afsis_soil_properties":
            train_name, test_name = "training.csv", "sorted_test.csv"
        if name == "unimelb_grant_application":
            train_name, test_name = "unimelb_training.csv", "unimelb_test.csv"
        if name == "mercari_price_suggestion_challenge":
            train_name, test_name, sep = "train.tsv", "test_stg2.tsv", "\t"
            test = test.rename(columns={"train_id": "test_id"})
        train.to_csv(output / train_name, index=False, sep=sep)
        test.to_csv(output / test_name, index=False, sep=sep)
    return frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=sorted(SPECS))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate(args.dataset, args.output, args.seed)
    print(f"Wrote synthetic {args.dataset} files to {args.output}")
