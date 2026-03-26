"""
Классификация конкурентов по ценовому сегменту: CatBoost Classifier.
Spec: specs/agents/analyst_agent.md — раздел «ML подход»

Классы: economy | standard | premium
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from src.models.research_report import Competitor

logger = logging.getLogger(__name__)

CONFIG_PATH = Path("src/ml/configs/competitor_classifier.json")
MODEL_PATH = Path("src/ml/competitor_classifier.cbm")

with CONFIG_PATH.open(encoding="utf-8") as f:
    _CFG = json.load(f)

FEATURES = _CFG["features"]
CLASSES = _CFG["classes"]
FEATURE_DESCRIPTIONS = _CFG["feature_descriptions"]


# ---------------------------------------------------------------------------
# Подготовка данных
# ---------------------------------------------------------------------------

def competitors_to_df(competitors: list[Competitor]) -> pd.DataFrame:
    rows = []
    for c in competitors:
        price_min = c.price_per_night_rub.min if c.price_per_night_rub else 0
        price_max = c.price_per_night_rub.max if c.price_per_night_rub else 0
        rows.append({
            "name": c.name,
            "distance_spb_km": c.distance_spb_km or 75.0,
            "price_per_night_min": price_min,
            "price_per_night_max": price_max,
            "cottage_count": c.cottage_count or 1,
            "rating": c.rating or 0.0,
            "reviews_count": c.reviews_count or 0,
            "infrastructure_count": len(c.infrastructure) if c.infrastructure else 0,
        })
    return pd.DataFrame(rows)


def _label_price_segment(price_max: float) -> str:
    """Экспертная разметка ценового сегмента по макс. цене за ночь."""
    if price_max < 5_000:
        return "economy"
    elif price_max < 15_000:
        return "standard"
    return "premium"


# ---------------------------------------------------------------------------
# CatBoost Classifier
# ---------------------------------------------------------------------------

def train(df: pd.DataFrame) -> CatBoostClassifier:
    str_target = df["price_per_night_max"].apply(_label_price_segment)
    # Кодируем строковые метки в целые числа
    label_to_int = {cls: i for i, cls in enumerate(CLASSES)}
    int_target = str_target.map(label_to_int)

    cfg = {k: v for k, v in _CFG["catboost"].items() if k not in ("loss_function", "eval_metric")}
    model = CatBoostClassifier(**cfg, loss_function="MultiClass", classes_count=len(CLASSES))
    model.fit(df[FEATURES], int_target)
    model.save_model(str(MODEL_PATH))
    logger.info("Classifier сохранён: %s", MODEL_PATH)
    return model


def load_or_train(df: pd.DataFrame) -> CatBoostClassifier:
    model = CatBoostClassifier()
    if MODEL_PATH.exists() and len(df) == 0:
        model.load_model(str(MODEL_PATH))
        logger.info("Classifier загружен из кэша")
    else:
        model = train(df)
    return model


def feature_importance_table(model: CatBoostClassifier) -> pd.DataFrame:
    imp = model.get_feature_importance()
    return pd.DataFrame({
        "feature": FEATURES,
        "importance_pct": imp.round(2),
        "description": [FEATURE_DESCRIPTIONS[f] for f in FEATURES],
    }).sort_values("importance_pct", ascending=False)


# ---------------------------------------------------------------------------
# Основная функция классификации
# ---------------------------------------------------------------------------

def classify_competitors(competitors: list[Competitor]) -> pd.DataFrame:
    """
    Принимает список Competitor, возвращает DataFrame с предсказанным
    ценовым сегментом и вероятностями классов.
    """
    if not competitors:
        return pd.DataFrame()

    df = competitors_to_df(competitors)
    model = load_or_train(df)

    raw_preds = model.predict(df[FEATURES]).flatten()
    probabilities = model.predict_proba(df[FEATURES])

    # Маппинг числовых меток обратно в строки
    label_map = {i: cls for i, cls in enumerate(CLASSES)}
    predictions = [label_map.get(int(p), str(p)) for p in raw_preds]

    result = df[["name", "distance_spb_km", "price_per_night_min",
                 "price_per_night_max", "rating"]].copy()
    result["segment"] = predictions
    for i, cls in enumerate(CLASSES):
        result[f"prob_{cls}"] = probabilities[:, i].round(3)

    imp_table = feature_importance_table(model)

    out_dir = Path("data/processed")
    out_dir.mkdir(exist_ok=True)
    result.to_csv(out_dir / "competitor_segments.csv", index=False)
    imp_table.to_csv(out_dir / "competitor_feature_importance.csv", index=False)

    logger.info("Классификация конкурентов:\n%s", result[["name", "segment"]].to_string(index=False))
    logger.info("Важность признаков:\n%s", imp_table.to_string(index=False))

    return result
