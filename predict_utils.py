import numpy as np
import pandas as pd
import re
import ast
from pathlib import Path

# ===== Config =====
SEED = 26
EMBEDDING_MODE = "none"
EMBEDDING_SAMPLE_K = 200
EMBEDDING_SVD_K = 200
EMB_PATH = Path("data") / "airbnb-use-embeddings.csv"


# ===== Helper functions =====

def parse_money_to_float(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)

    s = str(x).strip()
    if not s or s.lower() in {"na", "nan", "none", "null"}:
        return np.nan

    if s.endswith("%"):
        try:
            return float(s[:-1]) / 100.0
        except Exception:
            return np.nan

    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg = True
        s = s[1:-1].strip()

    s = re.sub(r"[\$,]", "", s)

    try:
        val = float(s)
        return -val if neg else val
    except Exception:
        return np.nan


def parse_percent_to_float(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if not s:
        return np.nan
    if s.endswith("%"):
        s = s[:-1]
    try:
        return float(s) / 100.0
    except Exception:
        return np.nan


def to_bool01(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, bool):
        return 1.0 if x else 0.0
    s = str(x).strip().lower()
    if s in {"t", "true", "1", "yes", "y"}:
        return 1.0
    if s in {"f", "false", "0", "no", "n"}:
        return 0.0
    return np.nan


def parse_listlike(x):
    if pd.isna(x):
        return []
    if isinstance(x, list):
        return [str(i).strip() for i in x if str(i).strip()]

    s = str(x).strip()
    if not s or s.lower() in {"na", "nan", "none", "null", "[]"}:
        return []

    try:
        v = ast.literal_eval(s)
        if isinstance(v, list):
            return [str(i).strip() for i in v if str(i).strip()]
    except Exception:
        pass

    s2 = s.strip("[]").replace("'", "").replace('"', "")
    parts = [p.strip() for p in s2.split(",") if p.strip()]
    return parts


def clean_zipcode(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    s = s.split("-")[0].split("\\")[0].strip()
    s = re.sub(r"\s+", "", s)
    return s[:5] if re.match(r"^\d{5}", s) else np.nan


def freq_encode_from_train(train_s, test_s):
    tr = train_s.fillna("NaN").astype(str)
    te = test_s.fillna("NaN").astype(str)
    freq = tr.value_counts(normalize=True)

    return (
        tr.map(freq).fillna(0).to_numpy().reshape(-1, 1),
        te.map(freq).fillna(0).to_numpy().reshape(-1, 1),
    )


# ===== Column groups =====

PRICE_COLUMNS = ["price", "weekly_price", "monthly_price"]

NUMERIC_COLS = [
    "host_listings_count",
    "host_total_listings_count",
    "accommodates",
    "bathrooms",
    "bedrooms",
    "guests_included",
    "minimum_nights",
    "maximum_nights",
    "availability_30",
    "availability_60",
    "availability_90",
    "availability_365",
    "number_of_reviews",
    "review_scores_rating",
    "review_scores_accuracy",
    "review_scores_cleanliness",
    "review_scores_checkin",
    "review_scores_communication",
    "review_scores_location",
    "review_scores_value",
    "host_tenure_days",
    "days_since_first_review",
    "days_since_last_review",
    "has_reviews",
    "host_response_rate",
    "host_acceptance_rate",
]

STRING_NUMERIC_COLS = ["beds", "security_deposit", "cleaning_fee", "extra_people"]

BOOL_COLS = [
    "require_guest_profile_picture",
    "require_guest_phone_verification",
    "instant_bookable",
    "is_business_travel_ready",
    "has_availability",
    "is_location_exact",
    "host_identity_verified",
    "host_has_profile_pic",
    "host_is_superhost",
]

CAT_COLS = [
    "property_type",
    "room_type",
    "bed_type",
    "cancellation_policy",
    "host_response_time",
]

SET_COL = "host_verifications"
LOC_COLS = ["latitude", "longitude"]
ZIP_COL = "zipcode"


# ===== Feature builder =====

def one_hot_from_train(train_s: pd.Series, test_s: pd.Series, prefix: str):
    train_s = train_s.fillna("NaN").astype(str)
    categories = pd.Index(train_s.unique())
    train_cat = pd.Categorical(train_s, categories=categories)

    test_s = test_s.fillna("NaN").astype(str)
    test_cat = pd.Categorical(test_s, categories=categories)

    train_oh = pd.get_dummies(train_cat, prefix=prefix)
    test_oh = pd.get_dummies(test_cat, prefix=prefix)
    test_oh = test_oh.reindex(columns=train_oh.columns, fill_value=0)

    return (
        train_oh.astype(np.float32).to_numpy(),
        test_oh.astype(np.float32).to_numpy(),
        list(train_oh.columns),
    )


def many_hot_from_train(train_lists, test_lists, prefix: str):
    vocab = sorted({t for row in train_lists for t in row})
    idx = {t: i for i, t in enumerate(vocab)}

    def build(lists):
        X = np.zeros((len(lists), len(vocab)), dtype=np.float32)
        for r, row in enumerate(lists):
            for t in row:
                i = idx.get(t)
                if i is not None:
                    X[r, i] = 1.0
        return X

    X_tr = build(train_lists)
    X_te = build(test_lists)
    names = [f"{prefix}__{t}" for t in vocab]
    return X_tr, X_te, names


def build_features(train_df: pd.DataFrame, test_df: pd.DataFrame):
    feats_tr = []
    feats_te = []
    feat_names = []

    # Numeric + missing indicators
    for col in NUMERIC_COLS:
        if col not in train_df.columns or col not in test_df.columns:
            continue

        tr = pd.to_numeric(train_df[col], errors="coerce")
        te = pd.to_numeric(test_df[col], errors="coerce")

        tr_missing = tr.isna().astype(np.float32).to_numpy().reshape(-1, 1)
        te_missing = te.isna().astype(np.float32).to_numpy().reshape(-1, 1)

        tr = tr.fillna(0.0).astype(np.float32).to_numpy().reshape(-1, 1)
        te = te.fillna(0.0).astype(np.float32).to_numpy().reshape(-1, 1)

        feats_tr.extend([tr, tr_missing])
        feats_te.extend([te, te_missing])
        feat_names.extend([col, f"{col}__missing"])

    # String numeric / money
    for col in STRING_NUMERIC_COLS:
        if col not in train_df.columns or col not in test_df.columns:
            continue

        tr_raw = train_df[col].apply(parse_money_to_float)
        te_raw = test_df[col].apply(parse_money_to_float)

        tr_missing = tr_raw.isna().astype(np.float32).to_numpy().reshape(-1, 1)
        te_missing = te_raw.isna().astype(np.float32).to_numpy().reshape(-1, 1)

        tr = tr_raw.fillna(0.0).astype(np.float32).to_numpy().reshape(-1, 1)
        te = te_raw.fillna(0.0).astype(np.float32).to_numpy().reshape(-1, 1)

        feats_tr.extend([tr, tr_missing])
        feats_te.extend([te, te_missing])
        feat_names.extend([col, f"{col}__missing"])

    # Boolean
    for col in BOOL_COLS:
        if col not in train_df.columns or col not in test_df.columns:
            continue

        tr_raw = train_df[col].apply(to_bool01)
        te_raw = test_df[col].apply(to_bool01)

        tr_missing = tr_raw.isna().astype(np.float32).to_numpy().reshape(-1, 1)
        te_missing = te_raw.isna().astype(np.float32).to_numpy().reshape(-1, 1)

        tr = tr_raw.fillna(0.0).astype(np.float32).to_numpy().reshape(-1, 1)
        te = te_raw.fillna(0.0).astype(np.float32).to_numpy().reshape(-1, 1)

        feats_tr.extend([tr, tr_missing])
        feats_te.extend([te, te_missing])
        feat_names.extend([col, f"{col}__missing"])

    # Categorical one-hot
    for col in CAT_COLS:
        if col not in train_df.columns or col not in test_df.columns:
            continue

        tr_oh, te_oh, names = one_hot_from_train(train_df[col], test_df[col], prefix=col)
        feats_tr.append(tr_oh)
        feats_te.append(te_oh)
        feat_names.extend(names)

    # Host verifications many-hot + count
    if SET_COL in train_df.columns and SET_COL in test_df.columns:
        tr_lists = train_df[SET_COL].apply(parse_listlike).tolist()
        te_lists = test_df[SET_COL].apply(parse_listlike).tolist()

        tr_mh, te_mh, names = many_hot_from_train(tr_lists, te_lists, prefix=SET_COL)
        feats_tr.append(tr_mh)
        feats_te.append(te_mh)
        feat_names.extend(names)

        feats_tr.append(np.array([len(x) for x in tr_lists], dtype=np.float32).reshape(-1, 1))
        feats_te.append(np.array([len(x) for x in te_lists], dtype=np.float32).reshape(-1, 1))
        feat_names.append(f"{SET_COL}__count")

    # Location polynomial features
    if all(c in train_df.columns and c in test_df.columns for c in LOC_COLS):
        tr_lat = pd.to_numeric(train_df["latitude"], errors="coerce").fillna(0.0).astype(np.float32).to_numpy()
        tr_lon = pd.to_numeric(train_df["longitude"], errors="coerce").fillna(0.0).astype(np.float32).to_numpy()
        te_lat = pd.to_numeric(test_df["latitude"], errors="coerce").fillna(0.0).astype(np.float32).to_numpy()
        te_lon = pd.to_numeric(test_df["longitude"], errors="coerce").fillna(0.0).astype(np.float32).to_numpy()

        tr_loc = np.vstack([tr_lat, tr_lon, tr_lat**2, tr_lon**2, tr_lat * tr_lon]).T
        te_loc = np.vstack([te_lat, te_lon, te_lat**2, te_lon**2, te_lat * te_lon]).T

        feats_tr.append(tr_loc)
        feats_te.append(te_loc)
        feat_names.extend(["latitude", "longitude", "latitude2", "longitude2", "lat_lon"])

    # Zipcode frequency encoding
    if ZIP_COL in train_df.columns and ZIP_COL in test_df.columns:
        tr_f, te_f = freq_encode_from_train(train_df[ZIP_COL], test_df[ZIP_COL])
        feats_tr.append(tr_f.astype(np.float32))
        feats_te.append(te_f.astype(np.float32))
        feat_names.append(f"{ZIP_COL}__freq")

    X_tr = np.hstack(feats_tr) if feats_tr else np.zeros((len(train_df), 0), dtype=np.float32)
    X_te = np.hstack(feats_te) if feats_te else np.zeros((len(test_df), 0), dtype=np.float32)

    return X_tr, X_te, feat_names