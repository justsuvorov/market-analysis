from __future__ import annotations

from pydantic import BaseModel, Field


class ResearchInput(BaseModel):
    region: str = Field(default="Ленинградская область")
    center_city: str = Field(default="Санкт-Петербург")
    radius_km: tuple[int, int] = Field(default=(50, 100))
    priority_location: str = Field(default="Сосновый Бор")
    alternative_locations: list[str] = Field(
        default=["Приморский район", "Курортный район", "побережье Ладоги"]
    )
    competitor_count: int = Field(default=12, ge=10, le=15)
    failed_cases_count: int = Field(default=2, ge=2)
    audience_segments: list[str] = Field(
        default=["investor", "buyer", "renter"]
    )


class RawResearchData(BaseModel):
    competitor_raw: list[dict] = Field(
        default_factory=list,
        description="Сырые данные конкурентов (Sutochno.ru, 101Hotels, Ozon Travel, Островок)",
    )
    reviews_raw: list[dict] = Field(
        default_factory=list,
        description="Отзывы (TripAdvisor, Яндекс.Карты, Google Maps)",
    )
    trends_raw: dict = Field(
        default_factory=dict,
        description="Поисковые тренды (Яндекс.Wordstat, Google Trends)",
    )
    social_raw: list[dict] = Field(
        default_factory=list,
        description="Данные из соцсетей (VK, Telegram, Instagram)",
    )
    traffic_raw: dict = Field(
        default_factory=dict,
        description="Трафик конкурентов (SimilarWeb, Metrica.Guru)",
    )
    location_data: list[dict] = Field(
        default_factory=list,
        description="Данные по локациям (Авито Недвижимость, открытые источники)",
    )
