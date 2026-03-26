"""
AnalystAgent — структурированный анализ и финальный отчёт.
Spec: specs/agents/analyst_agent.md

Агент возвращает Markdown-строку.
ML-модули (CatBoost, GLM) вызываются отдельно и встраиваются в отчёт.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from pydantic_ai import Agent, RunContext
from pydantic_ai.models import ModelSettings
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider

from src.ml.competitor_classifier import classify_competitors
from src.ml.location_scorer import score_locations
from src.models.inputs import RawResearchData
from src.models.research_report import Competitor, Format

load_dotenv()

logger = logging.getLogger(__name__)

_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
_MODEL_NAME = os.getenv("AI_MODEL_NAME", "gemini-2.5-flash-lite")
_TEMPERATURE = float(os.getenv("AI_TEMPERATURE", "0.7"))

PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def _save(key: str, data: dict) -> None:
    path = PROCESSED_DIR / f"{key}.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    logger.info("Saved: %s", path)


# ---------------------------------------------------------------------------
# ML: скоринг локаций и классификация конкурентов (вне агента)
# ---------------------------------------------------------------------------

RAW_LOCATIONS = [
    {"name": "Сосновый Бор", "distance_spb_min": 80, "land_price_rub_sotka": 180_000,
     "competition_density": 3, "water_proximity_km": 0.5,
     "unique_attraction": "Бухта Батарейная — спот для серфинга",
     "infrastructure_score": 6, "legal_restrictions": False},
    {"name": "Приморский район", "distance_spb_min": 40, "land_price_rub_sotka": 450_000,
     "competition_density": 12, "water_proximity_km": 1.2, "unique_attraction": None,
     "infrastructure_score": 9, "legal_restrictions": True,
     "legal_notes": "Ограничения ИЖС в прибрежной зоне"},
    {"name": "Курортный район", "distance_spb_min": 50, "land_price_rub_sotka": 380_000,
     "competition_density": 9, "water_proximity_km": 0.8, "unique_attraction": None,
     "infrastructure_score": 8, "legal_restrictions": False},
    {"name": "Побережье Ладоги", "distance_spb_min": 70, "land_price_rub_sotka": 120_000,
     "competition_density": 2, "water_proximity_km": 0.3,
     "unique_attraction": "Ладожское озеро, рыбалка, природные маршруты",
     "infrastructure_score": 4, "legal_restrictions": False},
]


def compute_ml_blocks() -> str:
    """Запускает ML-модели и возвращает Markdown-блок с результатами."""
    lines = []

    # Скоринг локаций
    scored = score_locations(RAW_LOCATIONS)
    _save("locations_scored", {"locations": [l.model_dump() for l in scored]})

    lines += ["\n## 3. Скоринг локаций (CatBoost + GLM)\n",
              "| Локация | Балл | Время от СПб | Земля (руб/сотка) | Рекомендация |",
              "|---|---|---|---|---|"]
    for loc in scored:
        lines.append(
            f"| **{loc.name}** | {loc.total_score:.1f}/10 | "
            f"{loc.distance_spb_min} мин | "
            f"{loc.land_price_rub_sotka:,} | "
            f"{(loc.recommendation or '').split('.')[0]} |"
        )

    # Классификация конкурентов
    stub_competitors = [
        Competitor(name="Ямилахти", location="Карелия", format=Format.eco_hotel,
                   is_failed_case=False),
        Competitor(name="Место силы", location="Дятлицы", format=Format.cottage_village,
                   is_failed_case=False),
    ]
    try:
        seg_df = classify_competitors(stub_competitors)
        if not seg_df.empty:
            lines += ["\n### Ценовые сегменты конкурентов\n",
                      "| Объект | Сегмент |", "|---|---|"]
            for _, row in seg_df.iterrows():
                lines.append(f"| {row['name']} | {row['segment']} |")
    except Exception as e:
        logger.warning("Classifier skipped: %s", e)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pydantic-AI Agent — output_type=str (Markdown)
# ---------------------------------------------------------------------------

_agent: Agent[RawResearchData, str] | None = None


def get_agent() -> Agent[RawResearchData, str]:
    global _agent
    if _agent is None:
        _agent = Agent(
            model=GoogleModel(_MODEL_NAME, provider=GoogleProvider(api_key=_API_KEY)),
            model_settings=ModelSettings(temperature=_TEMPERATURE),
            output_type=str,
            deps_type=RawResearchData,
            system_prompt=(
                "Ты старший маркетинговый аналитик рынка загородной недвижимости России. "
                "Анализируй данные о коттеджных комплексах Ленинградской области. "
                "Все выводы и рекомендации формулируй на русском языке. "
                "Отвечай только в формате Markdown."
            ),
        )

        @_agent.tool
        async def get_competitors_data(ctx: RunContext[RawResearchData]) -> str:
            """Вернуть данные о конкурентах из сырых результатов исследования."""
            raw = ctx.deps.competitor_raw
            if not raw:
                return "Данные о конкурентах не собраны (источники вернули 403/404)."
            return json.dumps(raw[:10], ensure_ascii=False, indent=2)

        @_agent.tool
        async def get_trends_data(ctx: RunContext[RawResearchData]) -> str:
            """Вернуть данные трендов."""
            return json.dumps(ctx.deps.trends_raw, ensure_ascii=False, indent=2)

        @_agent.tool
        async def get_location_scores(ctx: RunContext[RawResearchData]) -> str:
            """Получить результаты ML-скоринга локаций."""
            scored = await asyncio.to_thread(score_locations, RAW_LOCATIONS)
            summary = [
                f"{l.name}: {l.total_score:.1f}/10 — {l.recommendation}" for l in scored
            ]
            return "\n".join(summary)

    return _agent


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------

async def run(raw: RawResearchData | None = None) -> tuple[str, str]:
    """
    Запустить агента.
    Возвращает (markdown_ai_part, markdown_full_report).
    """
    if raw is None:
        raw = RawResearchData()

    prompt = (
        "Сформируй маркетинговый аналитический отчёт по коттеджному комплексу "
        "в Ленинградской области. Структура отчёта:\n\n"
        "# Маркетинговое исследование: Коттеджный комплекс, Ленобласть\n\n"
        "## 1. Обзор рынка\n"
        "## 2. Конкурентный анализ (используй инструмент get_competitors_data)\n"
        "## 4. Анализ целевой аудитории\n"
        "   - Инвесторы в строительство\n"
        "   - Покупатели домов\n"
        "   - Арендаторы\n"
        "## 5. Варианты концепции проекта (2-3 варианта с УТП)\n"
        "## 6. Вопросы для экспертных интервью\n"
        "## 7. SWOT-анализ\n"
        "## 8. Выводы и рекомендации\n\n"
        "Используй инструменты get_competitors_data, get_trends_data, get_location_scores. "
        "Пиши подробно, конкретно, на русском языке."
    )

    result = await get_agent().run(prompt, deps=raw)
    ai_markdown = result.output

    # Встраиваем ML-блок (раздел 3) между разделами 2 и 4
    ml_block = await asyncio.to_thread(compute_ml_blocks)

    # Вставляем ML после раздела 2
    if "## 4." in ai_markdown:
        full_report = ai_markdown.replace("## 4.", ml_block + "\n\n## 4.", 1)
    else:
        full_report = ai_markdown + "\n" + ml_block

    # Сохраняем
    report_path = PROCESSED_DIR / "research_report.md"
    report_path.write_text(full_report, encoding="utf-8")
    logger.info("Report saved: %s (%d chars)", report_path, len(full_report))

    return ai_markdown, full_report
