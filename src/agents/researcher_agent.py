"""
ResearcherAgent — сбор данных из веб-источников.
Spec: specs/agents/researcher_agent.md
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import os

import httpx
from dotenv import load_dotenv
from pydantic_ai import Agent, RunContext
from pydantic_ai.models import ModelSettings
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider

load_dotenv()

_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
_MODEL_NAME = os.getenv("AI_MODEL_NAME", "gemini-2.5-flash-lite")
_TEMPERATURE = float(os.getenv("AI_TEMPERATURE", "0.7"))

from src.models.inputs import RawResearchData, ResearchInput

logger = logging.getLogger(__name__)

CACHE_DIR = Path("data/raw")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

# Задержка между запросами (сек) для соблюдения rate limits
REQUEST_DELAY = 2.0


def _cache_path(key: str) -> Path:
    return CACHE_DIR / f"{key}.json"


def _load_cache(key: str) -> dict | list | None:
    path = _cache_path(key)
    if path.exists():
        logger.info("Cache hit: %s", key)
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def _save_cache(key: str, data: dict | list) -> None:
    _cache_path(key).write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("Cached: %s", key)


def _get(client: httpx.Client, url: str, params: dict | None = None) -> httpx.Response:
    logger.info("[%s] GET %s params=%s", datetime.utcnow().isoformat(), url, params)
    time.sleep(REQUEST_DELAY)
    return client.get(url, params=params, headers=HEADERS, timeout=15)


# ---------------------------------------------------------------------------
# Функции сбора данных
# ---------------------------------------------------------------------------

def fetch_sutochno(inp: ResearchInput, client: httpx.Client) -> list[dict]:
    """Поиск объектов на Sutochno.ru через публичный поиск."""
    cache_key = "sutochno_lenobl"
    if cached := _load_cache(cache_key):
        return cached

    results = []
    try:
        resp = _get(
            client,
            "https://sutochno.ru/api/search",
            params={
                "location": inp.region,
                "type": "house",
                "limit": 50,
            },
        )
        if resp.status_code == 200:
            data = resp.json()
            results = data.get("items", data) if isinstance(data, dict) else data
    except Exception as e:
        logger.warning("Sutochno fetch failed: %s", e)

    _save_cache(cache_key, results)
    return results


def fetch_avito_locations(inp: ResearchInput, client: httpx.Client) -> list[dict]:
    """Данные о земельных участках / объектах на Авито."""
    cache_key = "avito_lenobl_locations"
    if cached := _load_cache(cache_key):
        return cached

    results = []
    try:
        resp = _get(
            client,
            "https://www.avito.ru/api/9/items",
            params={
                "locationId": 141957,  # Ленинградская область
                "categoryId": 24,      # Загородная недвижимость
                "limit": 50,
            },
        )
        if resp.status_code == 200:
            data = resp.json()
            results = data.get("items", [])
    except Exception as e:
        logger.warning("Avito fetch failed: %s", e)

    _save_cache(cache_key, results)
    return results


def fetch_wordstat_trends(inp: ResearchInput, client: httpx.Client) -> dict:
    """Поисковые тренды — публичные данные Google Trends (RSS)."""
    cache_key = "trends_cottage_lenobl"
    if cached := _load_cache(cache_key):
        return cached

    queries = [
        "коттедж аренда Ленинградская область",
        "загородный дом аренда СПб",
        "коттеджный комплекс Сосновый Бор",
    ]
    results: dict = {"queries": queries, "fetched_at": datetime.utcnow().isoformat(), "data": {}}

    try:
        for q in queries:
            resp = _get(
                client,
                "https://trends.google.com/trends/api/dailytrends",
                params={"hl": "ru", "tz": "-180", "geo": "RU-LEN", "ns": 15},
            )
            results["data"][q] = {
                "status": resp.status_code,
                "available": resp.status_code == 200,
            }
    except Exception as e:
        logger.warning("Trends fetch failed: %s", e)

    _save_cache(cache_key, results)
    return results


def fetch_booking_competitors(inp: ResearchInput, client: httpx.Client) -> list[dict]:
    """Конкуренты через публичный поиск 101Hotels."""
    cache_key = "competitors_101hotels"
    if cached := _load_cache(cache_key):
        return cached

    results = []
    locations = [inp.priority_location] + inp.alternative_locations[:2]

    for loc in locations:
        try:
            resp = _get(
                client,
                "https://www.101hotels.ru/search",
                params={"query": f"коттедж {loc}", "type": "cottage"},
            )
            results.append({
                "source": "101hotels",
                "location": loc,
                "status": resp.status_code,
                "url": str(resp.url),
                "content_length": len(resp.content),
            })
        except Exception as e:
            logger.warning("101Hotels fetch failed for %s: %s", loc, e)

    _save_cache(cache_key, results)
    return results


# ---------------------------------------------------------------------------
# Pydantic-AI Agent (ленивая инициализация)
# ---------------------------------------------------------------------------

_agent: Agent[ResearchInput, RawResearchData] | None = None


def get_agent() -> Agent[ResearchInput, RawResearchData]:
    global _agent
    if _agent is None:
        _agent = Agent(
            model=GoogleModel(_MODEL_NAME, provider=GoogleProvider(api_key=_API_KEY)),
            model_settings=ModelSettings(temperature=_TEMPERATURE),
            output_type=RawResearchData,
            deps_type=ResearchInput,
            system_prompt=(
                "Ты агент-исследователь рынка загородной недвижимости Ленинградской области. "
                "Твоя задача — собрать максимально полные данные о конкурентах, трендах, "
                "локациях и отзывах из открытых веб-источников. "
                "Все текстовые поля заполняй на русском языке. "
                "Соблюдай rate limits источников."
            ),
        )

        @_agent.tool
        async def collect_competitors(ctx: RunContext[ResearchInput]) -> list[dict]:
            """Собрать данные о конкурентах с платформ бронирования."""
            with httpx.Client() as client:
                return await asyncio.to_thread(fetch_booking_competitors, ctx.deps, client)

        @_agent.tool
        async def collect_location_data(ctx: RunContext[ResearchInput]) -> list[dict]:
            """Собрать данные о локациях и земельных участках."""
            with httpx.Client() as client:
                return await asyncio.to_thread(fetch_avito_locations, ctx.deps, client)

        @_agent.tool
        async def collect_trends(ctx: RunContext[ResearchInput]) -> dict:
            """Собрать данные о поисковых трендах."""
            with httpx.Client() as client:
                return await asyncio.to_thread(fetch_wordstat_trends, ctx.deps, client)

        @_agent.tool
        async def collect_rental_listings(ctx: RunContext[ResearchInput]) -> list[dict]:
            """Собрать объявления аренды с Sutochno.ru."""
            with httpx.Client() as client:
                return await asyncio.to_thread(fetch_sutochno, ctx.deps, client)

    return _agent


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------

async def run(inp: ResearchInput | None = None) -> RawResearchData:
    """Запустить агента и вернуть сырые данные исследования."""
    if inp is None:
        inp = ResearchInput()
    result = await get_agent().run(
        "Собери все данные для маркетингового исследования.", deps=inp
    )
    return result.output
