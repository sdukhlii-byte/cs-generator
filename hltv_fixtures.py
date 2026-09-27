"""Автоматический подбор ближайших РЕАЛЬНЫХ матчей CS2 — аналог fixtures.py
из ai-match-lab, только источник — не football-data.org, а сама HLTV.

Страница https://www.hltv.org/matches отдаёт список предстоящих матчей
статикой: команды, турнир, unix-время начала и "звёзды значимости" матча
(0-5, чем больше — тем важнее событие, топ-старт почти всегда 4-5).
Именно эта страница и даёт нам match_url, который потом использует
cs_stats.lookup() — без него статы не собрать.

MIN_STARS фильтрует шум (квалификации категории D никому не интересны):
по умолчанию берём только матчи с 3+ звёздами.
"""

from __future__ import annotations

import datetime
import logging

import requests
from bs4 import BeautifulSoup

import state
from config import env_int

log = logging.getLogger("hltv_fixtures")

BASE = "https://www.hltv.org"
MATCHES_URL = f"{BASE}/matches"
HTTP_TIMEOUT = 20
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
    "Accept-Language": "en-US,en;q=0.9",
}


class NoFixturesFound(RuntimeError):
    """Нет ни одного подходящего матча в окне поиска — нормальное состояние
    (затишье между турнирами), а не сбой: вызывающий код должен тихо
    завершиться, а не падать и уходить в рестарт-луп."""


def _stars(el) -> int:
    return len(el.select(".stars .star")) if el else 0


def parse_matches_page(html: str, now: datetime.datetime | None = None) -> list[dict]:
    """[{"home","away","competition","date","match_url","id","stars"}], от
    ближайшего к самому дальнему. Отдельная функция от fetch() специально —
    чтобы тестировать разбор на статичном HTML без похода в сеть."""
    now = now or datetime.datetime.now(datetime.timezone.utc)
    soup = BeautifulSoup(html, "html.parser")
    out = []
    for card in soup.select("div.upcomingMatch, a.match"):
        href = card.get("href") or ""
        if not href:
            link = card.select_one("a")
            href = link.get("href") if link else ""
        if not href:
            continue
        teams = [t.get_text(strip=True) for t in card.select(".matchTeamName")]
        if len(teams) < 2:
            continue
        ts_el = card.select_one("[data-unix]")
        if not ts_el:
            continue
        try:
            unix_ms = int(ts_el["data-unix"])
        except (KeyError, ValueError):
            continue
        dt = datetime.datetime.fromtimestamp(unix_ms / 1000, tz=datetime.timezone.utc)
        event_el = card.select_one(".matchEventName, .matchInfoEmpty")
        match_id = href.strip("/").split("/")[1] if len(href.strip("/").split("/")) > 1 else href
        out.append({
            "home": teams[0],
            "away": teams[1],
            "competition": event_el.get_text(strip=True) if event_el else "",
            "date": dt.date().isoformat(),
            "match_url": BASE + href if href.startswith("/") else href,
            "id": f"hltv-{match_id}",
            "stars": _stars(card),
            "_dt": dt,
        })
    out.sort(key=lambda m: m["_dt"])
    return out


def fetch(days_ahead: int | None = None, per_run: int | None = None,
          min_stars: int | None = None) -> list[dict]:
    days_ahead = env_int("FIXTURES_DAYS_AHEAD", 7, lo=1, hi=30) if days_ahead is None else days_ahead
    per_run = env_int("FIXTURES_PER_RUN", 1, lo=1, hi=20) if per_run is None else per_run
    min_stars = env_int("MIN_STARS", 3, lo=0, hi=5) if min_stars is None else min_stars

    try:
        r = requests.get(MATCHES_URL, headers=HEADERS, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"Не скачал {MATCHES_URL}: {e}") from e

    now = datetime.datetime.now(datetime.timezone.utc)
    cutoff = now + datetime.timedelta(days=days_ahead)
    candidates = [
        m for m in parse_matches_page(r.text, now=now)
        if now <= m["_dt"] <= cutoff and m["stars"] >= min_stars
    ]
    if not candidates:
        raise NoFixturesFound(
            f"Нет матчей с {min_stars}+ звёздами в ближайшие {days_ahead} дн. — тихая пауза")

    keys = {m["id"] for m in candidates}
    already = state.already_posted(keys)
    fresh = [m for m in candidates if m["id"] not in already]
    if not fresh:
        raise NoFixturesFound("Все найденные матчи уже публиковались — жду новых")

    avoid_n = env_int("FIXTURES_AVOID_LAST_COMPETITIONS", 2, lo=0, hi=10)
    avoid = set(state.recent_competitions(avoid_n)) if avoid_n else set()
    ordered = sorted(fresh, key=lambda m: (m["competition"] in avoid, m["_dt"]))
    for m in ordered:
        m.pop("_dt", None)
    return ordered[:per_run]
