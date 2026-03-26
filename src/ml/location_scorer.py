"""
Скоринг локаций: CatBoost Regressor + GLM (statsmodels OLS).
Spec: specs/agents/analyst_agent.md — раздел «ML подход»

CatBoost — итоговый балл (target = инвестиционная привлекательность).
GLM       — интерпретация весов признаков для бизнес-отчёта.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from catboost import CatBoostRegressor

from src.models.research_report import Location

logger = logging.getLogger(__name__)

CONFIG_PATH = Path("src/ml/configs/location_scorer.json")
MODEL_PATH = Path("src/ml/location_scorer.cbm")

# Загружаем конфиг один раз при импорте
with CONFIG_PATH.open(encoding="utf-8") as f:
    _CFG = json.load(f)

FEATURES = _CFG["features"]
FEATURE_DESCRIPTIONS = _CFG["feature_descriptions"]


# ---------------------------------------------------------------------------
# Подготовка данных
# ---------------------------------------------------------------------------

def locations_to_df(locations: list[dict]) -> pd.DataFrame:
    """Привести список сырых локаций к DataFrame признаков."""
    rows = []
    for loc in locations:
        rows.append({
            "name": loc.get("name", ""),
            "distance_spb_min": loc.get("distance_spb_min", 60),
            "land_price_rub_sotka": loc.get("land_price_rub_sotka", 200_000),
            "competition_density": loc.get("competition_density", 5),
            "water_proximity_km": loc.get("water_proximity_km", 1.0),
            "has_unique_attraction": int(bool(loc.get("unique_attraction"))),
            "infrastructure_score": loc.get("infrastructure_score", 5),
            "legal_restrictions": int(loc.get("legal_restrictions", False)),
        })
    return pd.DataFrame(rows)


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Min-max нормализация признаков (0–1)."""
    X = df[FEATURES].copy().astype(float)
    for col in X.columns:
        mn, mx = X[col].min(), X[col].max()
        if mx > mn:
            X[col] = (X[col] - mn) / (mx - mn)
    return X


# ---------------------------------------------------------------------------
# Синтетический target для обучения на малых данных
# ---------------------------------------------------------------------------

def _synthetic_target(df: pd.DataFrame) -> pd.Series:
    """
    Экспертно взвешенный target для начального обучения.
    Знаки и веса объяснимы для бизнеса:
      - близость к СПб и воде увеличивают привлекательность
      - высокая конкуренция и правовые ограничения снижают
      - уникальный аттрактор и инфраструктура повышают
    """
    w = {
        "distance_spb_min":    -0.25,
        "land_price_rub_sotka": -0.15,
        "competition_density":  -0.20,
        "water_proximity_km":   -0.10,
        "has_unique_attraction": 0.20,
        "infrastructure_score":  0.20,
        "legal_restrictions":   -0.10,
    }
    X = _normalize(df)
    score = sum(X[col] * weight for col, weight in w.items())
    # Масштабируем в диапазон 1–10
    mn, mx = score.min(), score.max()
    if mx > mn:
        score = 1 + 9 * (score - mn) / (mx - mn)
    else:
        score = pd.Series([5.0] * len(score), index=score.index)
    return score.round(2)


# ---------------------------------------------------------------------------
# CatBoost
# ---------------------------------------------------------------------------

def train_catboost(df: pd.DataFrame, target: pd.Series) -> CatBoostRegressor:
    cfg = _CFG["catboost"]
    model = CatBoostRegressor(**cfg)
    model.fit(df[FEATURES], target)
    model.save_model(str(MODEL_PATH))
    logger.info("CatBoost модель сохранена: %s", MODEL_PATH)
    return model


def load_or_train_catboost(df: pd.DataFrame, target: pd.Series) -> CatBoostRegressor:
    model = CatBoostRegressor()
    if MODEL_PATH.exists():
        model.load_model(str(MODEL_PATH))
        logger.info("CatBoost модель загружена из кэша")
    else:
        model = train_catboost(df, target)
    return model


def feature_importance_table(model: CatBoostRegressor) -> pd.DataFrame:
    imp = model.get_feature_importance()
    df = pd.DataFrame({
        "feature": FEATURES,
        "importance_pct": imp.round(2),
        "description": [FEATURE_DESCRIPTIONS[f] for f in FEATURES],
    }).sort_values("importance_pct", ascending=False)
    return df


# ---------------------------------------------------------------------------
# GLM (statsmodels OLS) — интерпретируемые коэффициенты
# ---------------------------------------------------------------------------

def fit_glm(X_norm: pd.DataFrame, target: pd.Series) -> sm.regression.linear_model.RegressionResultsWrapper:
    X_with_const = sm.add_constant(X_norm[FEATURES])
    model = sm.OLS(target, X_with_const).fit()
    logger.info("GLM R²=%.3f", model.rsquared)
    return model


def glm_coefficient_table(result: sm.regression.linear_model.RegressionResultsWrapper) -> pd.DataFrame:
    """Таблица коэффициентов GLM для бизнес-отчёта."""
    coef_df = pd.DataFrame({
        "feature": result.params.index,
        "coefficient": result.params.values.round(4),
        "p_value": result.pvalues.values.round(4),
        "significant": result.pvalues.values < 0.05,
    })
    coef_df["description"] = coef_df["feature"].map(
        lambda f: FEATURE_DESCRIPTIONS.get(f, "константа")
    )
    return coef_df.sort_values("coefficient", key=abs, ascending=False)


# ---------------------------------------------------------------------------
# Основная функция скоринга
# ---------------------------------------------------------------------------

def score_locations(locations: list[dict]) -> list[Location]:
    """
    Принимает сырые данные локаций, возвращает список Location
    с заполненными total_score и recommendation.
    """
    if not locations:
        return []

    df = locations_to_df(locations)
    target = _synthetic_target(df)
    X_norm = _normalize(df)

    # CatBoost
    cb_model = load_or_train_catboost(df, target)
    cb_scores = cb_model.predict(df[FEATURES])

    # GLM
    glm_result = fit_glm(X_norm, target)
    imp_table = feature_importance_table(cb_model)
    coef_table = glm_coefficient_table(glm_result)

    logger.info("\nВажность признаков (CatBoost):\n%s", imp_table.to_string(index=False))
    logger.info("\nКоэффициенты GLM:\n%s", coef_table.to_string(index=False))

    # Сохраняем таблицы для отчёта
    out_dir = Path("data/processed")
    out_dir.mkdir(exist_ok=True)
    imp_table.to_csv(out_dir / "feature_importance.csv", index=False)
    coef_table.to_csv(out_dir / "glm_coefficients.csv", index=False)

    # Формируем рекомендации
    scored: list[Location] = []
    max_score = cb_scores.max()

    for i, raw_loc in enumerate(locations):
        score = round(float(cb_scores[i]), 2)
        is_best = score == round(float(max_score), 2)

        if score >= 7.5:
            rec = "Приоритетная локация — высокая инвестиционная привлекательность."
        elif score >= 5.0:
            rec = "Перспективная локация — требует дополнительной проработки."
        else:
            rec = "Низкий приоритет — значительные ограничения."

        if is_best:
            rec = "[TOP] РЕКОМЕНДУЕТСЯ. " + rec

        scored.append(Location(
            name=raw_loc.get("name", f"Локация {i+1}"),
            distance_spb_min=raw_loc.get("distance_spb_min", 60),
            land_price_rub_sotka=raw_loc.get("land_price_rub_sotka"),
            competition_density=raw_loc.get("competition_density"),
            water_proximity_km=raw_loc.get("water_proximity_km"),
            unique_attraction=raw_loc.get("unique_attraction"),
            infrastructure_score=raw_loc.get("infrastructure_score", 5),
            legal_restrictions=bool(raw_loc.get("legal_restrictions", False)),
            legal_notes=raw_loc.get("legal_notes"),
            total_score=score,
            recommendation=rec,
        ))

    return sorted(scored, key=lambda loc: loc.total_score or 0, reverse=True)
