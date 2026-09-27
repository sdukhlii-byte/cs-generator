"""Реальная статистика по матчу CS2 — с HLTV.

Аналог stats.py из ai-match-lab (там — Elo/форма/H2H по CSV с football-data.co.uk),
только источник другой и, что важнее, ДОСТУПНОСТЬ данных другая.

Матч-страница HLTV (просто GET, без кликов и JS) отдаёт статикой:
  - названия команд и их текущий мировой рейтинг (# в топе);
  - счёт личных встреч (H2H) этих двух команд;
  - последние результаты (winstreak) обеих команд;
  - ссылки на профили команд.
Профиль команды (тоже просто GET по ссылке с матч-страницы) даёт:
  - "недель вместе" текущим составом (стабильность ростера);
  - средний возраст состава;
  - если отрисовано сервером — winrate по картам за последние 3 месяца.

ЧЕГО ТУТ НЕТ и почему: у HLTV есть более глубокая аналитика по матчу —
5v4/4v5 клатчи, % пистолетных раундов, split раундов атака/защита по карте,
% первого фрага. Она подгружается JS-виджетом по клику на "Match analytics"
и не присутствует в исходном HTML. Открытый скрейпер cs_bot (см. review),
на основе которого списан этот модуль, добывает её через Selenium с кликами
по вкладкам — но даже там на каждое поле есть `except: <захардкоженная
заглушка>` (age.append(23.0), sum_3m += 0.86 и т.д.), то есть даже с полным
браузером это ненадёжно. Раз даже авторский скрейпер держит фиктивные
дефолты для этих полей, нет смысла тащить сюда риск Selenium (реальный
браузер, куки, гораздо больше шансов словить бан IP от Cloudflare на
проде-крон-джобе) ради данных, которые и так наполовину придуманы.
Если позже понадобится — этот модуль легко расширить, добавив реального
Playwright-скрейпера для "Match analytics"; сигнатура format_for_prompt()
не изменится, просто в data появятся новые честные поля.

Если HLTV блокирует запрос (антибот, смена вёрстки) — lookup() возвращает
пустой словарь, а не бросает исключение: как и в ai-match-lab, отсутствие
статы не должно ронять генерацию всего матча, модели просто прогнозируют
без неё (это лучше, чем совсем не выпустить ролик).
"""

from __future__ import annotations

import logging
import re

import requests
from bs4 import BeautifulSoup

log = logging.getLogger("cs_stats")

BASE = "https://www.hltv.org"
HTTP_TIMEOUT = 20
# HLTV режет запросы без правдоподобного браузерного User-Agent почти сразу.
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
    "Accept-Language": "en-US,en;q=0.9",
}


def _get(url: str) -> BeautifulSoup | None:
    try:
        r = requests.get(url, headers=HEADERS, timeout=HTTP_TIMEOUT)
        if r.status_code == 403:
            log.warning("HLTV отдал 403 на %s — похоже на антибот-блок", url)
            return None
        r.raise_for_status()
        return BeautifulSoup(r.text, "html.parser")
    except requests.RequestException as e:
        log.warning("Не скачал %s: %s", url, e)
        return None


def _int_or_none(text: str) -> int | None:
    m = re.search(r"\d+", text or "")
    return int(m.group()) if m else None


def _float_or_none(text: str) -> float | None:
    m = re.search(r"\d+(?:\.\d+)?", (text or "").replace(",", "."))
    return float(m.group()) if m else None


def parse_match_page(soup: BeautifulSoup) -> dict:
    """То, что реально лежит в статике страницы матча — без единого клика."""
    out: dict = {}

    names = [t.get_text(strip=True) for t in soup.select(".standard-box.teamsBox .teamName")]
    if len(names) >= 2:
        out["home_name"], out["away_name"] = names[0], names[1]

    ranks = soup.select(".teamRanking")
    if len(ranks) >= 2:
        out["home_rank"] = _int_or_none(ranks[0].get_text())
        out["away_rank"] = _int_or_none(ranks[1].get_text())

    h2h_right = soup.select_one(".flexbox-column.flexbox-center.grow.right-border .bold")
    h2h_left = soup.select_one(".flexbox-column.flexbox-center.grow.left-border .bold")
    if h2h_right and h2h_left:
        out["home_h2h_wins"] = _int_or_none(h2h_right.get_text())
        out["away_h2h_wins"] = _int_or_none(h2h_left.get_text())

    streak_boxes = soup.select(".past-matches-box.text-ellipsis")[2:4]
    streaks = []
    for box in streak_boxes:
        el = box.select_one(".past-matches-streak")
        streaks.append(el.get_text(strip=True) if el else "")
    if len(streaks) == 2:
        out["home_streak"], out["away_streak"] = streaks[0], streaks[1]

    # Ссылки на профили команд — идём по ним обычным GET, кликать не нужно.
    team_links = soup.select("a.team")
    hrefs = [a.get("href") for a in team_links if a.get("href")]
    if len(hrefs) >= 2:
        out["home_profile_url"] = BASE + hrefs[0] if hrefs[0].startswith("/") else hrefs[0]
        out["away_profile_url"] = BASE + hrefs[1] if hrefs[1].startswith("/") else hrefs[1]

    return out


def parse_team_profile(soup: BeautifulSoup) -> dict:
    """Профиль команды: недели вместе, средний возраст — тоже статика."""
    out: dict = {}
    stats = soup.select(".profile-team-stat")
    for row in stats:
        name_el = row.select_one(".right")
        label = row.get_text(" ", strip=True).lower()
        if not name_el:
            continue
        val = name_el.get_text(strip=True)
        if "week" in label:
            out["weeks_together"] = _float_or_none(val)
        elif "age" in label:
            out["avg_age"] = _float_or_none(val)
        elif "world rank" in label:
            out["world_rank"] = _int_or_none(val)

    # Логотип команды — обычно og:image на странице профиля.
    meta = soup.select_one('meta[property="og:image"]')
    if meta and meta.get("content"):
        out["logo_url"] = meta["content"]
    return out


def lookup(home: str, away: str, match_url: str = "") -> dict:
    """Собирает всё, что доступно, для пары команд.

    match_url — прямая ссылка на страницу матча на HLTV (её лучше всего
    брать из hltv_fixtures.fetch(), который и так её видел при сборе
    расписания). Без неё честный live-поиск команды по названию не входит
    в v1 — просто возвращаем пустой словарь, промпт уйдёт без статы.
    """
    if not match_url:
        log.info("Нет match_url для %s vs %s — иду без статы HLTV", home, away)
        return {}

    soup = _get(match_url)
    if soup is None:
        return {}
    data = parse_match_page(soup)

    for side, key in (("home", "home_profile_url"), ("away", "away_profile_url")):
        url = data.get(key)
        if not url:
            continue
        prof = _get(url)
        if prof is None:
            continue
        for k, v in parse_team_profile(prof).items():
            data[f"{side}_{k}"] = v

    return data


def format_for_prompt(data: dict, home: str, away: str) -> str:
    """Только реально найденные поля — ничего не выдумываем на пропуски."""
    if not data:
        return ""
    lines = []
    if data.get("home_rank") or data.get("away_rank"):
        lines.append(f'HLTV world ranking: {home} #{data.get("home_rank", "?")}, '
                     f'{away} #{data.get("away_rank", "?")}')
    if data.get("home_h2h_wins") is not None and data.get("away_h2h_wins") is not None:
        lines.append(f'Head-to-head wins: {home} {data["home_h2h_wins"]} — '
                     f'{away} {data["away_h2h_wins"]}')
    if data.get("home_streak") or data.get("away_streak"):
        lines.append(f'Recent form: {home} {data.get("home_streak", "n/a")}, '
                     f'{away} {data.get("away_streak", "n/a")}')
    if data.get("home_weeks_together") or data.get("away_weeks_together"):
        lines.append(f'Roster stability (weeks together): {home} '
                     f'{data.get("home_weeks_together", "n/a")}, '
                     f'{away} {data.get("away_weeks_together", "n/a")}')
    if data.get("home_avg_age") or data.get("away_avg_age"):
        lines.append(f'Average squad age: {home} {data.get("home_avg_age", "n/a")}, '
                     f'{away} {data.get("away_avg_age", "n/a")}')
    return "\n".join(lines)
