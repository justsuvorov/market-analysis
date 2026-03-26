"""
Точка входа: маркетинговое исследование коттеджного комплекса, Ленобласть.
Запуск: python main.py
"""

import asyncio
import sys
from pathlib import Path

# Добавляем корень проекта в PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent))

from src.agents.pipeline import main

if __name__ == "__main__":
    asyncio.run(main())
