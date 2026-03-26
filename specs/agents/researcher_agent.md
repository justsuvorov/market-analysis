# Спецификация: ResearcherAgent

## Status
`Draft`

## Purpose
Автономный агент для сбора данных из веб-источников.
Собирает информацию о конкурентах, локациях, трендах и отзывах
для последующей передачи в `AnalystAgent`.

## Input
**Модель:** `ResearchInput` (см. `specs/marketing_research_v1.md`)

## Output
**Модель:** `RawResearchData`

| Поле | Тип | Источник |
|---|---|---|
| `competitor_raw` | `list[dict]` | Sutochno.ru, 101Hotels, Ozon Travel, Островок |
| `reviews_raw` | `list[dict]` | TripAdvisor, Яндекс.Карты, Google Maps |
| `trends_raw` | `dict` | Яндекс.Wordstat, Google Trends |
| `social_raw` | `list[dict]` | VK, Telegram, Instagram |
| `traffic_raw` | `dict` | SimilarWeb, Metrica.Guru |
| `location_data` | `list[dict]` | Открытые данные, Авито Недвижимость |

## Tools
- HTTP-клиент (httpx / aiohttp) для парсинга открытых источников
- Playwright (headless) для JS-рендеринга (Яндекс.Карты, отзывы)
- Pydantic-AI `Agent` с типизированным `result_type=RawResearchData`

## Constraints
- Соблюдать robots.txt и rate limits источников
- Кэшировать результаты в `data/raw/` (JSON) для повторного использования
- Логировать все запросы с timestamp и источником

## Решения
- HTTP-клиент: httpx (без Playwright)
- Авторизация: только публичные endpoints, без авторизации
