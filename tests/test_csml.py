"""Тесты cs-match-lab: только то, что реально своё (счёт по картам, HLTV-парсинг,
CS2-обвязка generate.py). poster.py/video.py/config.py/state.py перенесены из
ai-match-lab без изменений — их отдельный тест-сьют уже покрывает движок, тут
только пара smoke-тестов, что интеграция не сломалась."""

from __future__ import annotations

import datetime
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bs4 import BeautifulSoup

import cs_stats
import generate
import hltv_fixtures
import poster
import predictions
import video

# ------------------------------------------------------------- predictions --


def test_valid_map_score_rejects_draw():
    assert predictions._valid_map_score(1, 1, 2) is False


def test_valid_map_score_rejects_out_of_range():
    assert predictions._valid_map_score(3, 1, 2) is False
    assert predictions._valid_map_score(2, 2, 2) is False  # тоже ничья


def test_valid_map_score_requires_winner_to_hit_target():
    assert predictions._valid_map_score(1, 0, 2) is False  # ни один не дошёл до 2
    assert predictions._valid_map_score(2, 0, 2) is True
    assert predictions._valid_map_score(2, 1, 2) is True


def test_parse_bo3_from_json():
    h, a, why = predictions._parse('{"home_maps": 2, "away_maps": 1, "reason": "map pool edge"}', 2)
    assert (h, a, why) == (2, 1, "map pool edge")


def test_parse_bo5_allows_up_to_three():
    h, a, _ = predictions._parse('{"home_maps": 3, "away_maps": 2, "reason": "close series"}', 3)
    assert (h, a) == (3, 2)


def test_parse_bo1_only_allows_1_0_or_0_1():
    h, a, _ = predictions._parse('{"home_maps": 1, "away_maps": 0, "reason": "one map"}', 1)
    assert (h, a) == (1, 0)
    try:
        predictions._parse('{"home_maps": 1, "away_maps": 1, "reason": "impossible"}', 1)
        assert False, "Bo1 не может закончиться 1-1"
    except predictions.ModelError:
        pass


def test_parse_rejects_draw_json_falls_back_to_regex():
    # ничья в JSON отбрасывается, но валидный счёт "2-1" в тексте — подхватывается.
    h, a, _ = predictions._parse(
        'thinking... {"home_maps": 1, "away_maps": 1} anyway I predict 2-1', 2)
    assert (h, a) == (2, 1)


def test_parse_rejects_pure_nonsense():
    try:
        predictions._parse("no score here at all", 2)
        assert False
    except predictions.ModelError:
        pass


def test_build_prompt_bo_label_and_win_maps():
    prompt = predictions._build_prompt({"home": "NAVI", "away": "G2", "bo": 5}, "")
    assert "Bo5" in prompt
    assert "0-3" in prompt  # win_maps для Bo5 — 3


def test_build_prompt_defaults_to_bo3_when_missing():
    prompt = predictions._build_prompt({"home": "A", "away": "B"}, "")
    assert "Bo3" in prompt
    assert "0-2" in prompt


def test_build_prompt_ignores_unknown_match_keys():
    # лишние ключи матча (home_logo, scores, match_url, ...) не должны падать format()
    prompt = predictions._build_prompt(
        {"home": "A", "away": "B", "home_logo": "x", "match_url": "y", "scores": "2-0"}, "")
    assert "A" in prompt and "B" in prompt


# ------------------------------------------------------------------ cs_stats --


def _match_page_html(future=True):
    return """
    <div class="standard-box teamsBox">
      <div class="teamName">Natus Vincere</div>
      <div class="teamName">G2</div>
    </div>
    <div class="teamRanking">#3</div>
    <div class="teamRanking">#5</div>
    <div class="flexbox-column flexbox-center grow right-border"><div class="bold">7</div></div>
    <div class="flexbox-column flexbox-center grow left-border"><div class="bold">3</div></div>
    <div class="past-matches-box text-ellipsis">x</div>
    <div class="past-matches-box text-ellipsis">x</div>
    <div class="past-matches-box text-ellipsis"><div class="past-matches-streak">W3</div></div>
    <div class="past-matches-box text-ellipsis"><div class="past-matches-streak">L1</div></div>
    <a class="team" href="/team/4608/natus-vincere">NAVI</a>
    <a class="team" href="/team/5995/g2">G2</a>
    """


def test_parse_match_page_extracts_ranking_and_h2h():
    soup = BeautifulSoup(_match_page_html(), "html.parser")
    data = cs_stats.parse_match_page(soup)
    assert data["home_rank"] == 3 and data["away_rank"] == 5
    assert data["home_h2h_wins"] == 7 and data["away_h2h_wins"] == 3
    assert data["home_streak"] == "W3" and data["away_streak"] == "L1"
    assert data["home_profile_url"].endswith("/team/4608/natus-vincere")


def test_parse_match_page_tolerates_missing_sections():
    # HLTV меняет вёрстку — отсутствие блока не должно ронять разбор.
    soup = BeautifulSoup("<div>ничего интересного</div>", "html.parser")
    data = cs_stats.parse_match_page(soup)
    assert data == {}


def test_parse_team_profile_reads_weeks_age_logo():
    html = """
    <meta property="og:image" content="https://img.hltv.org/logo/navi.png">
    <div class="profile-team-stat">World rank: <span class="right">#3</span></div>
    <div class="profile-team-stat">Weeks in current lineup: <span class="right">42</span></div>
    <div class="profile-team-stat">Average player age: <span class="right">24.5</span></div>
    """
    soup = BeautifulSoup(html, "html.parser")
    data = cs_stats.parse_team_profile(soup)
    assert data["weeks_together"] == 42
    assert data["avg_age"] == 24.5
    assert data["world_rank"] == 3
    assert data["logo_url"] == "https://img.hltv.org/logo/navi.png"


def test_format_for_prompt_only_includes_real_fields():
    block = cs_stats.format_for_prompt(
        {"home_rank": 3, "away_rank": 5}, "Natus Vincere", "G2")
    assert "HLTV world ranking" in block
    assert "Head-to-head" not in block  # не выдумываем то, чего не нашли


def test_format_for_prompt_empty_data_is_empty_string():
    assert cs_stats.format_for_prompt({}, "A", "B") == ""


def test_lookup_without_match_url_returns_empty(monkeypatch):
    # v1 честно не ищет команду по имени без прямой ссылки на матч.
    assert cs_stats.lookup("Natus Vincere", "G2", "") == {}


# --------------------------------------------------------------- hltv_fixtures --


def _matches_page_html(offset_days=2, stars=4):
    now = datetime.datetime(2026, 9, 27, tzinfo=datetime.timezone.utc)
    ts = int((now + datetime.timedelta(days=offset_days)).timestamp() * 1000)
    star_divs = '<div class="star"></div>' * stars
    return now, f"""
    <div class="upcomingMatch">
      <a href="/matches/12345/navi-vs-g2">
        <div class="matchTeamName">Natus Vincere</div>
        <div class="matchTeamName">G2</div>
        <div data-unix="{ts}"></div>
        <div class="matchEventName">IEM Katowice 2026</div>
        <div class="stars">{star_divs}</div>
      </a>
    </div>
    """


def test_parse_matches_page_extracts_match():
    now, html = _matches_page_html()
    matches = hltv_fixtures.parse_matches_page(html, now=now)
    assert len(matches) == 1
    m = matches[0]
    assert m["home"] == "Natus Vincere" and m["away"] == "G2"
    assert m["competition"] == "IEM Katowice 2026"
    assert m["match_url"] == "https://www.hltv.org/matches/12345/navi-vs-g2"
    assert m["id"] == "hltv-12345"
    assert m["stars"] == 4


def test_parse_matches_page_sorted_soonest_first():
    now, html_far = _matches_page_html(offset_days=5)
    _, html_near = _matches_page_html(offset_days=1)
    combined = html_far + html_near
    matches = hltv_fixtures.parse_matches_page(combined, now=now)
    assert len(matches) == 2
    assert matches[0]["date"] < matches[1]["date"]


def test_parse_matches_page_skips_cards_without_teams():
    html = '<div class="upcomingMatch"><a href="/matches/1/x">no teams here</a></div>'
    assert hltv_fixtures.parse_matches_page(html) == []


def test_fetch_raises_no_fixtures_when_below_star_threshold(monkeypatch):
    now, html = _matches_page_html(stars=1)

    class FakeResp:
        text = html
        def raise_for_status(self):
            pass

    monkeypatch.setattr(hltv_fixtures._scraper, "get", lambda *a, **k: FakeResp())
    monkeypatch.setattr(hltv_fixtures.state, "already_posted", lambda keys: set())
    try:
        hltv_fixtures.fetch(days_ahead=7, per_run=1, min_stars=3)
        assert False, "1-звёздочный матч не должен пройти MIN_STARS=3"
    except hltv_fixtures.NoFixturesFound:
        pass


def test_fetch_skips_already_posted(monkeypatch):
    now, html = _matches_page_html(stars=5)

    class FakeResp:
        text = html
        def raise_for_status(self):
            pass

    monkeypatch.setattr(hltv_fixtures._scraper, "get", lambda *a, **k: FakeResp())
    monkeypatch.setattr(hltv_fixtures.state, "already_posted", lambda keys: keys)
    monkeypatch.setattr(hltv_fixtures.state, "recent_competitions", lambda n: [])
    try:
        hltv_fixtures.fetch(days_ahead=30, per_run=1, min_stars=0)
        assert False
    except hltv_fixtures.NoFixturesFound:
        pass


# ------------------------------------------------------------------- generate --


def test_slugify_reused_from_engine():
    assert generate.slugify("Natus Vincere-vs-G2-2026-10-19") == "natus-vincere-vs-g2-2026-10-19"


def test_captions_mention_bo_and_no_football_emoji():
    rows = [{"label": "ChatGPT", "home": 2, "away": 1, "reason": ""}]
    match = {"home": "NAVI", "away": "G2", "competition": "IEM", "date": "2026-10-19", "bo": 3}
    caps = generate.captions(match, rows)
    assert "Bo3" in caps["threads"]
    assert "⚽" not in caps["threads"]


def test_load_logo_falls_back_to_placeholder(tmp_path):
    img = generate.load_logo("", "Natus Vincere")
    assert img.size == (600, 400)


def test_run_end_to_end_with_manual_scores(tmp_path):
    """Полный прогон без сети и без видео: ровно то, чем реально пользуется
    --scores + --no-video + --no-send. Если движок (poster.py) сломается на
    Row(home=<maps>, away=<maps>) — тут это всплывёт."""
    import argparse

    match = {
        "home": "Natus Vincere", "away": "G2", "competition": "IEM Katowice 2026",
        "date": "2026-10-19", "bo": 3, "scores": "2-0,2-1,2-0,1-2,2-1",
        "home_logo": "", "away_logo": "",
    }
    args = argparse.Namespace(out=str(tmp_path), no_video=True, no_send=True, keep_temp=False)
    zpath = generate.run(match, args)
    assert os.path.exists(zpath)
    assert os.path.exists(os.path.join(tmp_path, "natus-vincere-vs-g2-2026-10-19", "filled.jpg"))


# ---------------------------------------------------------- poster/video smoke --


def test_poster_row_accepts_map_counts_not_just_goals():
    m = poster.Match(
        home="NAVI", away="G2",
        home_flag=generate._placeholder("NAVI"), away_flag=generate._placeholder("G2"),
        rows=[poster.Row("ChatGPT", 2, 0, "chatgpt")],
        consensus="NAVI", consensus_note="1 of 1 models agree", consensus_pct=100,
    )
    img = poster.render_paper(m, 1)
    assert img.size == (poster.PAPER_W, poster.PAPER_H)


def test_video_prompt_still_under_length_limit_for_cs2_rows():
    rows = [{"label": f"Model{i}", "home": 2, "away": 1, "icon": "chatgpt"} for i in range(5)]
    prompt = video.build_prompt(rows, 0, 5)
    assert len(prompt) <= 2500
