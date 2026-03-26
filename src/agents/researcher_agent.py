"""
ResearcherAgent — сбор данных из веб-источников.
Spec: specs/agents/researcher_agent.md

Стратегия: seed-данные + HTML-парсинг 101Hotels + Google Trends RSS.
При недоступности источника возвращает seed с пометкой source="seed".
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
import os

import httpx
from bs4 import BeautifulSoup
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
    ),
    "Accept-Language": "ru-RU,ru;q=0.9",
}

REQUEST_DELAY = 2.0

# ---------------------------------------------------------------------------
# Seed-данные: 12 известных конкурентов Ленобласти
# ---------------------------------------------------------------------------

SEED_COMPETITORS: list[dict] = [
    {"name": "Ямилахти", "location": "Карелия, Лахденпохья", "format": "eco_hotel",
     "price_per_night_rub": 18000, "rating": 4.7, "reviews_count": 312,
     "unique_features": ["финская баня", "выход к озеру", "глэмпинг"],
     "is_failed_case": False, "source": "seed"},
    {"name": "Место силы", "location": "Дятлицы, Ленобласть", "format": "cottage_village",
     "price_per_night_rub": 12000, "rating": 4.5, "reviews_count": 187,
     "unique_features": ["экотуризм", "медитации", "органическая ферма"],
     "is_failed_case": False, "source": "seed"},
    {"name": "Охта-парк", "location": "Лодейное поле, Ленобласть", "format": "resort",
     "price_per_night_rub": 22000, "rating": 4.6, "reviews_count": 524,
     "unique_features": ["горнолыжный склон", "аквапарк", "конный клуб"],
     "is_failed_case": False, "source": "seed"},
    {"name": "Коркки", "location": "Карелия, Сортавала", "format": "eco_hotel",
     "price_per_night_rub": 9500, "rating": 4.3, "reviews_count": 98,
     "unique_features": ["деревянные коттеджи", "Ладожское озеро"],
     "is_failed_case": False, "source": "seed"},
    {"name": "Изумрудный берег", "location": "Рощино, Ленобласть", "format": "cottage_village",
     "price_per_night_rub": 8500, "rating": 4.1, "reviews_count": 143,
     "unique_features": ["Финский залив", "сосновый лес"],
     "is_failed_case": False, "source": "seed"},
    {"name": "Репино Коттедж", "location": "Репино, Курортный р-н", "format": "cottage_village",
     "price_per_night_rub": 15000, "rating": 4.4, "reviews_count": 221,
     "unique_features": ["близость СПб 40 км", "залив", "ресторан"],
     "is_failed_case": False, "source": "seed"},
    {"name": "Зеркальный", "location": "Зеркальное, Ленобласть", "format": "country_club",
     "price_per_night_rub": 7000, "rating": 3.9, "reviews_count": 76,
     "unique_features": ["бывший советский лагерь", "озеро"],
     "is_failed_case": True, "source": "seed",
     "failure_reason": "Устаревшая инфраструктура, низкий рейтинг, отток клиентов"},
    {"name": "Финский залив SPA", "location": "Сосновый Бор", "format": "resort",
     "price_per_night_rub": 11000, "rating": 4.2, "reviews_count": 165,
     "unique_features": ["SPA-центр", "залив", "детская площадка"],
     "is_failed_case": False, "source": "seed"},
    {"name": "Приморье", "location": "Приморск, Ленобласть", "format": "cottage_village",
     "price_per_night_rub": 6500, "rating": 3.8, "reviews_count": 52,
     "unique_features": ["тихое место", "рыбалка"],
     "is_failed_case": True, "source": "seed",
     "failure_reason": "Слабый маркетинг, нет уникального предложения, закрылся в 2023"},
    {"name": "Ладога-клуб", "location": "Ладожское озеро, Кировский р-н", "format": "eco_hotel",
     "price_per_night_rub": 13500, "rating": 4.5, "reviews_count": 289,
     "unique_features": ["Ладожские шхеры", "рыбалка", "сапсёрфинг"],
     "is_failed_case": False, "source": "seed"},
    {"name": "Дом у леса", "location": "Всеволожск, Ленобласть", "format": "cottage_village",
     "price_per_night_rub": 9000, "rating": 4.3, "reviews_count": 134,
     "unique_features": ["30 мин от СПб", "лесные прогулки"],
     "is_failed_case": False, "source": "seed"},
    {"name": "Усадьба Орловка", "location": "Гатчинский р-н", "format": "country_club",
     "price_per_night_rub": 16000, "rating": 4.6, "reviews_count": 198,
     "unique_features": ["историческая усадьба", "конный клуб", "охота"],
     "is_failed_case": False, "source": "seed"},
]


# ---------------------------------------------------------------------------
# Утилиты кэша
# ---------------------------------------------------------------------------

def _cache_path(key: str) -> Path:
    return CACHE_DIR / f"{key}.json"


def _load_cache(key: str) -> dict | list | None:
    path = _cache_path(key)
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            # Не возвращать пустые кэши — они устарели
            if data:
                logger.info("Cache hit: %s", key)
                return data
        except Exception:
            pass
    return None


def _save_cache(key: str, data: dict | list) -> None:
    _cache_path(key).write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("Cached: %s", key)


def _get(client: httpx.Client, url: str, params: dict | None = None) -> httpx.Response:
    logger.info("[%s] GET %s params=%s", datetime.utcnow().isoformat(), url, params)
    time.sleep(REQUEST_DELAY)
    return client.get(url, params=params, headers=HEADERS, timeout=15, follow_redirects=True)


# ---------------------------------------------------------------------------
# Функции сбора данных
# ---------------------------------------------------------------------------

def fetch_competitors(inp: ResearchInput, client: httpx.Client) -> list[dict]:
    """
    Возвращает seed-список + данные с 101Hotels (если доступен).
    Дедуплицирует по name.
    """
    cache_key = "competitors_v2"
    if cached := _load_cache(cache_key):
        return cached

    results: list[dict] = list(SEED_COMPETITORS)
    seen_names = {c["name"].lower() for c in results}

    locations = [inp.priority_location] + inp.alternative_locations[:2]
    for loc in locations:
        try:
            resp = _get(
                client,
                "https://www.101hotels.ru/search",
                params={"query": f"коттедж {loc}"},
            )
            if resp.status_code == 200 and len(resp.content) > 500:
                soup = BeautifulSoup(resp.content, "lxml")
                # Ищем карточки объектов (разные возможные селекторы)
                for card in soup.select(".b-result-item, .hotel-card, [class*='result-item'], [class*='hotel-item']")[:5]:
                    name_el = card.select_one("[class*='name'], [class*='title'], h2, h3")
                    price_el = card.select_one("[class*='price'], [class*='cost']")
                    rating_el = card.select_one("[class*='rating'], [class*='score']")

                    name = name_el.get_text(strip=True) if name_el else None
                    if not name or name.lower() in seen_names:
                        continue

                    price_text = price_el.get_text(strip=True) if price_el else ""
                    price = None
                    for part in price_text.replace("\u00a0", "").split():
                        digits = "".join(c for c in part if c.isdigit())
                        if digits and len(digits) >= 3:
                            price = int(digits)
                            break

                    rating_text = rating_el.get_text(strip=True) if rating_el else ""
                    rating = None
                    for part in rating_text.replace(",", ".").split():
                        try:
                            v = float(part)
                            if 1.0 <= v <= 10.0:
                                rating = round(min(v, 5.0), 1)
                                break
                        except ValueError:
                            pass

                    entry = {
                        "name": name,
                        "location": loc,
                        "format": "cottage_village",
                        "price_per_night_rub": price,
                        "rating": rating,
                        "is_failed_case": False,
                        "source": "101hotels",
                    }
                    results.append(entry)
                    seen_names.add(name.lower())
                    logger.info("Parsed from 101hotels: %s", name)
        except Exception as e:
            logger.warning("101Hotels fetch failed for %s: %s", loc, e)

    _save_cache(cache_key, results)
    return results


def fetch_avito_locations(inp: ResearchInput, client: httpx.Client) -> list[dict]:
    """Данные о локациях — статические данные + попытка Avito."""
    cache_key = "avito_lenobl_locations"
    if cached := _load_cache(cache_key):
        return cached

    # Статические данные локаций как fallback
    static_locations = [
        {"name": "Сосновый Бор", "region": "Ленобласть",
         "avg_land_price_rub_sotka": 180000, "source": "seed"},
        {"name": "Рощино", "region": "Ленобласть",
         "avg_land_price_rub_sotka": 220000, "source": "seed"},
        {"name": "Репино", "region": "Курортный р-н СПб",
         "avg_land_price_rub_sotka": 450000, "source": "seed"},
        {"name": "Всеволожск", "region": "Ленобласть",
         "avg_land_price_rub_sotka": 280000, "source": "seed"},
        {"name": "Лодейное поле", "region": "Ленобласть",
         "avg_land_price_rub_sotka": 95000, "source": "seed"},
    ]

    results = list(static_locations)
    try:
        resp = _get(
            client,
            "https://www.avito.ru/api/9/items",
            params={"locationId": 141957, "categoryId": 24, "limit": 20},
        )
        if resp.status_code == 200:
            data = resp.json()
            items = data.get("items", [])
            for item in items[:10]:
                results.append({
                    "name": item.get("title", ""),
                    "region": "Ленобласть",
                    "price": item.get("price", {}).get("value"),
                    "source": "avito",
                })
    except Exception as e:
        logger.warning("Avito fetch failed: %s", e)

    _save_cache(cache_key, results)
    return results


def fetch_trends(inp: ResearchInput, client: httpx.Client) -> dict:
    """Google Trends через публичный RSS-фид (Россия)."""
    cache_key = "trends_cottage_lenobl_v2"
    if cached := _load_cache(cache_key):
        return cached

    results: dict = {
        "fetched_at": datetime.utcnow().isoformat(),
        "trending_topics": [],
        "cottage_keywords": [
            "коттедж аренда Ленинградская область",
            "загородный дом аренда СПб",
            "коттеджный комплекс Сосновый Бор",
        ],
        "market_signals": {
            "growing_segments": ["глэмпинг", "эко-отдых", "семейный отдых за городом"],
            "peak_season": "июнь-август",
            "weekend_demand": "высокий — пятница/суббота до 70% бронирований",
            "source": "seed_analysis",
        },
    }

    try:
        resp = _get(
            client,
            "https://trends.google.com/trends/hottrends/atom/feed",
            params={"pn": "p154"},  # Россия
        )
        if resp.status_code == 200:
            root = ET.fromstring(resp.content)
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            topics = []
            for entry in root.findall("atom:entry", ns)[:10]:
                title_el = entry.find("atom:title", ns)
                if title_el is not None and title_el.text:
                    topics.append(title_el.text)
            if topics:
                results["trending_topics"] = topics
                logger.info("Google Trends RSS: %d topics fetched", len(topics))
    except Exception as e:
        logger.warning("Trends RSS fetch failed: %s", e)

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
                "локациях и отзывах. Используй доступные инструменты. "
                "Все текстовые поля заполняй на русском языке."
            ),
        )

        @_agent.tool
        async def collect_competitors(ctx: RunContext[ResearchInput]) -> list[dict]:
            """Собрать данные о конкурентах (seed + HTML-парсинг 101Hotels)."""
            with httpx.Client() as client:
                return await asyncio.to_thread(fetch_competitors, ctx.deps, client)

        @_agent.tool
        async def collect_location_data(ctx: RunContext[ResearchInput]) -> list[dict]:
            """Собрать данные о локациях и земельных участках."""
            with httpx.Client() as client:
                return await asyncio.to_thread(fetch_avito_locations, ctx.deps, client)

        @_agent.tool
        async def collect_trends(ctx: RunContext[ResearchInput]) -> dict:
            """Собрать данные о поисковых трендах (Google Trends RSS)."""
            with httpx.Client() as client:
                return await asyncio.to_thread(fetch_trends, ctx.deps, client)

    return _agent


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------

async def run(inp: ResearchInput | None = None) -> RawResearchData:
    """
    Собрать сырые данные исследования.
    Данные собираются напрямую (без LLM-оркестрации) — надёжнее и быстрее.
    """
    if inp is None:
        inp = ResearchInput()

    with httpx.Client() as client:
        competitors = await asyncio.to_thread(fetch_competitors, inp, client)
        locations = await asyncio.to_thread(fetch_avito_locations, inp, client)
        trends = await asyncio.to_thread(fetch_trends, inp, client)

    logger.info(
        "Данные собраны: конкуренты=%d, локации=%d, тренды_топиков=%d",
        len(competitors),
        len(locations),
        len(trends.get("trending_topics", [])),
    )

    return RawResearchData(
        competitor_raw=competitors,
        location_data=locations,
        trends_raw=trends,
    )
