"""
Pipeline: ResearcherAgent → AnalystAgent
Запуск: python -m src.agents.pipeline
"""

from __future__ import annotations

import asyncio
import logging

from src.agents.analyst_agent import run as analyst_run
from src.agents.researcher_agent import ResearchInput, run as researcher_run

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


async def main() -> None:
    logger.info("=== Шаг 1: ResearcherAgent ===")
    raw_data = await researcher_run(ResearchInput())
    logger.info(
        "Собрано: конкуренты=%d, локации=%d, тренды=%s",
        len(raw_data.competitor_raw),
        len(raw_data.location_data),
        bool(raw_data.trends_raw),
    )

    logger.info("=== Шаг 2: AnalystAgent ===")
    _, full_report = await analyst_run(raw_data)
    logger.info("Отчёт сформирован: %d символов", len(full_report))

    print("\n" + "=" * 60)
    print(full_report[:3000])
    if len(full_report) > 3000:
        print(f"\n... (полный отчёт в data/processed/research_report.md)")


if __name__ == "__main__":
    asyncio.run(main())
