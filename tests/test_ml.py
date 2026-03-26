"""Тест ML-модулей без API ключа."""
import pandas as pd
from src.ml.location_scorer import score_locations
from src.ml.competitor_classifier import classify_competitors
from src.models.research_report import Competitor, Format, PricePerNightRub

LOCATIONS = [
    {"name": "Сосновый Бор", "distance_spb_min": 80, "land_price_rub_sotka": 180_000,
     "competition_density": 3, "water_proximity_km": 0.5,
     "unique_attraction": "серфинг", "infrastructure_score": 6, "legal_restrictions": False},
    {"name": "Приморский район", "distance_spb_min": 40, "land_price_rub_sotka": 450_000,
     "competition_density": 12, "water_proximity_km": 1.2,
     "unique_attraction": None, "infrastructure_score": 9, "legal_restrictions": True},
    {"name": "Курортный район", "distance_spb_min": 50, "land_price_rub_sotka": 380_000,
     "competition_density": 9, "water_proximity_km": 0.8,
     "unique_attraction": None, "infrastructure_score": 8, "legal_restrictions": False},
    {"name": "Побережье Ладоги", "distance_spb_min": 70, "land_price_rub_sotka": 120_000,
     "competition_density": 2, "water_proximity_km": 0.3,
     "unique_attraction": "рыбалка", "infrastructure_score": 4, "legal_restrictions": False},
]

COMPETITORS = [
    Competitor(name="Ямилахти", location="Карелия", format=Format.eco_hotel,
               is_failed_case=False, price_per_night_rub=PricePerNightRub(min=8000, max=20000),
               distance_spb_km=120, rating=4.7, reviews_count=340, cottage_count=15),
    Competitor(name="Место силы", location="Дятлицы", format=Format.cottage_village,
               is_failed_case=False, price_per_night_rub=PricePerNightRub(min=3000, max=6000),
               distance_spb_km=55, rating=4.2, reviews_count=89, cottage_count=8),
    Competitor(name="Провальный кейс", location="Выборг", format=Format.resort,
               is_failed_case=True, price_per_night_rub=PricePerNightRub(min=1500, max=3000),
               distance_spb_km=140, rating=2.1, reviews_count=12, cottage_count=4),
]


def test_location_scoring():
    results = score_locations(LOCATIONS)
    assert len(results) == 4
    scores = [r.total_score for r in results]
    assert all(1 <= s <= 10 for s in scores), f"Баллы вне диапазона: {scores}"
    # Отсортированы по убыванию
    assert scores == sorted(scores, reverse=True)
    print("Скоринг локаций:")
    for loc in results:
        print(f"  {loc.name}: {loc.total_score}")


def test_competitor_classification():
    df = classify_competitors(COMPETITORS)
    assert isinstance(df, pd.DataFrame)
    assert "segment" in df.columns
    assert len(df) == len(COMPETITORS)
    print("Классификация конкурентов:")
    print(df[["name", "segment"]].to_string(index=False))


if __name__ == "__main__":
    test_location_scoring()
    print()
    test_competitor_classification()
    print("\nВсе тесты прошли успешно.")
