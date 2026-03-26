# Маркетинговое исследование: Коттеджный комплекс, Ленобласть

Система автономных AI-агентов для проведения маркетингового исследования коттеджного комплекса с гостиничным управлением в Ленинградской области.

## Описание

Проект реализован по методологии **Spec-Driven Development (SDD)** — любое изменение в коде начинается с обновления спецификации в `/specs`.

Два агента на базе [Pydantic-AI](https://ai.pydantic.dev/) работают последовательно:

```
ResearcherAgent → AnalystAgent → research_report.md
```

- **ResearcherAgent** — собирает данные из открытых веб-источников (платформы бронирования, отзывы, тренды)
- **AnalystAgent** — формирует аналитический отчёт: рынок, конкуренты, локации, ЦА, концепция, SWOT

ML-модели (CatBoost + GLM) оценивают локации и классифицируют конкурентов по ценовому сегменту.

## Быстрый старт

### 1. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 2. Настройка окружения

```bash
cp .env.example .env
```

Открой `.env` и вставь свой Google Gemini API key:

```
GOOGLE_API_KEY=твой_ключ
AI_MODEL_NAME=gemini-2.5-flash-lite
AI_TEMPERATURE=0.7
```

Ключ можно получить на [aistudio.google.com](https://aistudio.google.com/app/apikey).

### 3. Запуск

```bash
python main.py
```

Финальный отчёт сохраняется в `data/processed/research_report.md`.

## Структура проекта

```
market-analysis/
├── main.py                         # Точка входа
├── requirements.txt
├── .env.example
│
├── specs/                          # Спецификации (источник правды)
│   ├── marketing_research_v1.md    # Главная спецификация
│   ├── agents/
│   │   ├── researcher_agent.md
│   │   └── analyst_agent.md
│   └── schemas/                    # JSON Schema
│       ├── competitor.json
│       ├── location.json
│       ├── customer_segment.json
│       └── research_report.json
│
├── src/
│   ├── agents/
│   │   ├── researcher_agent.py     # Сбор данных из веб-источников
│   │   ├── analyst_agent.py        # Анализ и генерация отчёта
│   │   └── pipeline.py             # Оркестрация агентов
│   ├── models/                     # Pydantic-модели (сгенерированы из схем)
│   │   ├── competitor.py
│   │   ├── location.py
│   │   ├── customer_segment.py
│   │   ├── research_report.py
│   │   └── inputs.py
│   └── ml/
│       ├── location_scorer.py      # CatBoost Regressor + GLM
│       ├── competitor_classifier.py # CatBoost Classifier
│       └── configs/                # Гиперпараметры моделей
│
├── data/
│   ├── raw/                        # Кэш сырых данных (gitignored)
│   └── processed/                  # Результаты анализа (gitignored)
│
└── tests/
    └── test_ml.py                  # Тесты ML-модулей
```

## Веб-источники

| Категория | Источники |
|---|---|
| Бронирование | Sutochno.ru, 101Hotels, Ozon Travel, Островок |
| Тренды | Яндекс.Wordstat, Google Trends |
| Отзывы | TripAdvisor, Яндекс.Карты, Google Maps |
| Недвижимость | Авито Недвижимость |
| Трафик | SimilarWeb |

## ML-модели

| Модель | Задача | Алгоритм |
|---|---|---|
| `location_scorer` | Скоринг локаций (1–10) | CatBoost Regressor + GLM |
| `competitor_classifier` | Ценовой сегмент конкурентов | CatBoost Classifier |

GLM используется для интерпретации весов признаков в бизнес-отчёте.

## Технологии

| Слой | Инструмент |
|---|---|
| AI-агенты | Pydantic-AI |
| LLM | Google Gemini (2.5-flash-lite) |
| Данные | Pydantic v2 |
| ML | CatBoost, statsmodels |
| HTTP | httpx |
| Генерация моделей | datamodel-codegen |

## Регенерация Pydantic-моделей

После изменения JSON Schema:

```bash
datamodel-codegen \
  --input specs/schemas/competitor.json \
  --input-file-type jsonschema \
  --output src/models/competitor.py \
  --use-annotated --field-constraints \
  --output-model-type pydantic_v2.BaseModel
```

## Тесты

```bash
PYTHONPATH=. python tests/test_ml.py
```

## Исходное ТЗ

`Коттеджный_комплекс_ТЗ_на_маркетинговое_исследование_Финал.docx`

Проект охватывает анализ рынка Ленобласти в радиусе 50–100 км от Санкт-Петербурга, приоритетная локация — Сосновый Бор.
