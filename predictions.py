"""Реальные прогнозы счёта по картам (CS2) от 5 моделей через OpenRouter.

Портировано из ai-match-lab: там модели предсказывали голы в футболе, тут —
счёт по картам в Bo{1,3,5} матче CS2 (home_maps/away_maps вместо
home_goals/away_goals). Вся инфраструктура вокруг (слоты, ретраи, разбор
JSON, ThreadPoolExecutor) один в один — сорт независим от предметной
области.

Каждую модель спрашиваем отдельно и одинаково: состав, статы с HLTV, формат
серии -> строгий JSON со счётом. Для моделей без встроенного поиска
включается веб-плагин OpenRouter (WEB_SEARCH=true), чтобы учитывались
свежие трансферы состава, форма на LAN vs online, недавние результаты.
Perplexity ищет в вебе сама.

ID моделей у OpenRouter быстро устаревают, поэтому у каждого слота есть
«предпочтительный» id и префикс-фоллбэк: если id пропал из каталога,
берётся самая свежая модель вендора с этим префиксом.

Отказ одной модели больше не роняет весь матч: слоты опрашиваются
независимо, а в конце проверяется, что успешных ответов хватает
(MIN_MODELS, по умолчанию все 5).
"""

from __future__ import annotations

import datetime
import json
import logging
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor

import requests

import cs_stats
from config import env_bool, env_int, env_str, require_env

log = logging.getLogger("predict")

OPENROUTER = "https://openrouter.ai/api/v1"
REQUEST_TIMEOUT = 180
ATTEMPTS = 3

# (подпись на бланке, ключ иконки, env-переменная, id по умолчанию, префикс-фоллбэк)
SLOTS = [
    ("ChatGPT", "chatgpt", "MODEL_CHATGPT", "openai/gpt-5", "openai/gpt-"),
    ("Claude", "claude", "MODEL_CLAUDE", "anthropic/claude-sonnet-4.5", "anthropic/claude-"),
    ("Gemini", "gemini", "MODEL_GEMINI", "google/gemini-2.5-pro", "google/gemini-"),
    ("Perplexity", "perplexity", "MODEL_PERPLEXITY", "perplexity/sonar-pro", "perplexity/sonar"),
    ("Grok", "grok", "MODEL_GROK", "x-ai/grok-4", "x-ai/grok-"),
]

# "-mini", а не "mini": иначе отсеется "gemini"
_SKIP = ("image", "audio", "vision-preview", "embed", "tts", "batch",
         ":free", "-mini", "-nano", "-lite")

PROMPT = """You are a Counter-Strike 2 esports analyst. Predict the exact map score of this UPCOMING {bo_label} match — it has not been played yet:

Today's date: {today}
{home} vs {away}
Event: {competition}
Match date: {date}
{stats_block}
Consider the stats above (if given) together with current roster/lineup news, recent
form (LAN vs online), map pool strengths and weaknesses, and head-to-head history.
Weigh recent roster changes or a stand-in over older stats when they conflict.
The score must be a valid {bo_label} result: one side wins with {win_maps} maps, the
other has strictly fewer. A draw is impossible.
Answer ONLY with JSON, no markdown:
{{"home_maps": <int 0-{win_maps}>, "away_maps": <int 0-{win_maps}>, "reason": "<one short sentence, max 20 words>"}}"""

# bo -> (label, maps needed to win). Bo2 не бывает в плей-офф (может быть
# ничья 1-1), поэтому не поддерживаем — не наш формат для однозначного счёта.
_BO_WIN = {1: 1, 3: 2, 5: 3}

# Reasoning-модели (gpt-5, gemini-2.5-pro, grok с thinking) тратят часть
# max_tokens на внутренние рассуждения — при низком лимите на сам JSON-ответ
# ничего не остаётся (пустой content) или он обрезается на середине.
# Даём большой запас и просим минимум размышлений — нам нужен только счёт.
_REASONING = {"effort": "low", "exclude": True}


class ModelError(RuntimeError):
    """Слот не смог выдать счёт. `fatal=True` — повторять запрос бессмысленно."""

    def __init__(self, message: str, fatal: bool = False):
        super().__init__(message)
        self.fatal = fatal


def _headers() -> dict:
    key = require_env(
        "OPENROUTER_API_KEY",
        "Ключ OpenRouter нужен для прогнозов. Либо задай его, либо запусти "
        'с --scores "1-2,2-1,2-2,1-0,2-1", чтобы обойтись без API.')
    return {
        "Authorization": f"Bearer {key}",
        "HTTP-Referer": env_str("OPENROUTER_REFERER", "https://t.me/aimatchlab"),
        "X-Title": "AI Match Lab",
    }


def _catalog() -> list:
    try:
        r = requests.get(f"{OPENROUTER}/models", timeout=30)
        r.raise_for_status()
        data = r.json().get("data", [])
        return [m for m in data if isinstance(m, dict) and m.get("id")]
    except Exception as e:  # каталог — необязательная роскошь, без него просто берём id как есть
        log.warning("Каталог OpenRouter недоступен (%s) — беру id как есть", e)
        return []


def resolve_models() -> list[tuple[str, str, str]]:
    """[(label, icon, model_id)] с проверкой по каталогу OpenRouter."""
    cat = _catalog()
    ids = {m["id"] for m in cat}
    out = []
    for label, icon, env, default, prefix in SLOTS:
        want = env_str(env) or default
        if not cat or want in ids:
            out.append((label, icon, want))
            continue
        cands = [m for m in cat
                 if m["id"].startswith(prefix) and not any(s in m["id"] for s in _SKIP)]
        # "created" у некоторых записей бывает null — сравнение None с int падало
        cands.sort(key=lambda m: m.get("created") or 0, reverse=True)
        if cands:
            log.info("%s: %s нет в каталоге, беру свежую %s", label, want, cands[0]["id"])
            out.append((label, icon, cands[0]["id"]))
        else:
            log.warning("%s: не нашёл ни %s, ни модели с префиксом %s", label, want, prefix)
            out.append((label, icon, want))
    return out


# ---------------------------------------------------------------- разбор ---

def _json_objects(text: str):
    """Все сбалансированные {...} в тексте, от последнего к первому.

    Старый `re.search(r"\\{.*?\\}")` брал ПЕРВУЮ пару скобок: если модель
    сначала писала рассуждение с фигурными скобками или markdown-пример,
    парсился мусор вместо настоящего ответа.
    """
    starts = [i for i, ch in enumerate(text) if ch == "{"]
    found = []
    for start in starts:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    found.append(text[start:i + 1])
                    break
    return list(reversed(found))


def _valid_map_score(h: int, a: int, win_maps: int) -> bool:
    """Валидный счёт Bo{2*win_maps-1}: обе стороны в диапазоне, ничьей нет,
    победитель набрал ровно win_maps, проигравший — строго меньше."""
    if not (0 <= h <= win_maps and 0 <= a <= win_maps):
        return False
    if h == a:
        return False
    return max(h, a) == win_maps


def _parse(text: str, win_maps: int = 2) -> tuple[int, int, str]:
    text = (text or "").strip()
    for blob in _json_objects(text):
        try:
            data = json.loads(blob)
            h, a = int(data["home_maps"]), int(data["away_maps"])
        except (ValueError, TypeError, KeyError, json.JSONDecodeError):
            continue
        if _valid_map_score(h, a, win_maps):
            return h, a, str(data.get("reason", ""))[:160].strip()
    # JSON мог обрезаться по max_tokens ещё до закрывающей скобки —
    # вытаскиваем поля по отдельности, не дожидаясь валидного объекта.
    hm = re.search(r'"home_maps"\s*:\s*(\d)', text)
    am = re.search(r'"away_maps"\s*:\s*(\d)', text)
    if hm and am and _valid_map_score(int(hm.group(1)), int(am.group(1)), win_maps):
        return int(hm.group(1)), int(am.group(1)), ""
    # совсем без JSON: первое «2-1» / «2:0» в ответе, что проходит валидацию
    for m in re.finditer(r"\b(\d)\s*[-:–]\s*(\d)\b", text):
        h, a = int(m.group(1)), int(m.group(2))
        if _valid_map_score(h, a, win_maps):
            return h, a, ""
    raise ModelError(f"не распарсил счёт по картам: {text[:200]!r}")


def _build_prompt(match: dict, stats_block: str) -> str:
    """format() только по известным полям: лишние ключи матча (home_logo,
    scores, ...) и коллизии имён вроде match['today'] больше не ломают вызов."""
    bo = match.get("bo") or 3
    win_maps = _BO_WIN.get(bo, 2)
    return PROMPT.format(
        today=datetime.date.today().isoformat(),
        home=match.get("home", ""),
        away=match.get("away", ""),
        competition=match.get("competition") or "n/a",
        date=match.get("date") or "n/a",
        bo_label=f"Bo{bo}",
        win_maps=win_maps,
        stats_block=f"\nReal stats:\n{stats_block}\n" if stats_block else "\n",
    )


# ------------------------------------------------------------------ запрос --

def _fatal_status(code: int) -> bool:
    """401/403 — плохой ключ, 400/404 — плохой запрос или несуществующая модель.
    Повторять такое три раза бессмысленно, только тратим время прогона."""
    return code in (400, 401, 403, 404)


def ask(model: str, match: dict, web: bool, stats_block: str) -> tuple[int, int, str]:
    win_maps = _BO_WIN.get(match.get("bo") or 3, 2)
    prompt = _build_prompt(match, stats_block)
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": env_int("MAX_TOKENS", 1500, lo=200, hi=8000),
        "reasoning": _REASONING,
    }
    # 0.4 был слишком «жадным»: с одинаковым промптом и одной и той же
    # реальной статистикой на входе разные модели регулярно сходились на
    # ОДНОМ И ТОМ ЖЕ счёте (например все пять — «3-1» для явного фаворита),
    # хотя причины в reason у каждой были свои. Это не баг пайплайна (каждая
    # модель реально опрашивается отдельно), а следствие низкой температуры:
    # при явном фаворите самый «типичный» счёт для такой истории один и тот
    # же у всех моделей. Выше температура — честнее показывает реальный
    # разброс мнений моделей, а не совпадение по одной и той же жадной моде.
    temperature = env_str("TEMPERATURE", "0.75")
    if temperature.lower() not in ("", "none", "off"):
        try:
            body["temperature"] = float(temperature.replace(",", "."))
        except ValueError:
            body["temperature"] = 0.75
    if web and not model.startswith("perplexity/"):
        body["plugins"] = [{"id": "web", "max_results": 4}]

    last: Exception | None = None
    for attempt in range(1, ATTEMPTS + 1):
        try:
            r = requests.post(f"{OPENROUTER}/chat/completions", headers=_headers(),
                              json=body, timeout=REQUEST_TIMEOUT)
            if r.status_code >= 400:
                raise ModelError(f"{r.status_code}: {r.text[:300]}",
                                 fatal=_fatal_status(r.status_code))
            payload = r.json()
            choices = payload.get("choices") or []
            if not choices:
                raise ModelError(f"пустой ответ без choices: {str(payload)[:200]}")
            content = (choices[0].get("message") or {}).get("content") or ""
            return _parse(content, win_maps)
        except ModelError as e:
            if e.fatal:
                log.error("%s: %s — повторять бессмысленно", model, e)
                raise
            last = e
        except (requests.RequestException, ValueError) as e:
            last = e
        log.warning("%s: попытка %d/%d не удалась — %s", model, attempt, ATTEMPTS, last)
        if attempt < ATTEMPTS:
            # экспоненциальная пауза с джиттером: без неё три попытки
            # укладывались в одну секунду и упирались в тот же rate limit
            time.sleep(min(2 ** attempt, 8) + random.uniform(0, 0.8))
    raise ModelError(f"{model}: {last}")


# --------------------------------------------------------------- пайплайн ---

def collect_stats_block(match: dict) -> str:
    if not env_bool("USE_STATS", True):
        return ""
    try:
        real = cs_stats.lookup(match["home"], match["away"], match.get("match_url", ""))
        block = cs_stats.format_for_prompt(real, match["home"], match["away"])
        if block:
            log.info("Статистика найдена (%s vs %s):\n%s", match["home"], match["away"], block)
        return block
    except Exception as e:
        log.warning("Не удалось получить статистику: %s", e)
        return ""


def predict_all(match: dict) -> list[dict]:
    """
    match: {home, away, competition, date}
    -> [{"label","icon","model","home","away","reason"}] в порядке SLOTS.
    """
    web = env_bool("WEB_SEARCH", True)
    models = resolve_models()
    stats_block = collect_stats_block(match)
    min_models = env_int("MIN_MODELS", len(models), lo=1, hi=len(models))

    def one(slot):
        label, icon, model = slot
        try:
            h, a, why = ask(model, match, web, stats_block)
        except Exception as e:
            log.error("%-10s %s — не ответила: %s", label, model, e)
            return label, None, e
        log.info("%-10s %s  %d-%d  %s", label, model, h, a, why)
        return label, {"label": label, "icon": icon, "model": model,
                       "home": h, "away": a, "reason": why}, None

    # Слоты независимы: раньше исключение внутри ex.map() обрывало весь
    # список, и падение одной модели стоило целого матча.
    with ThreadPoolExecutor(max_workers=max(1, len(models))) as ex:
        results = list(ex.map(one, models))

    rows = [row for _, row, _ in results if row]
    failures = [(label, err) for label, row, err in results if not row]
    if len(rows) < min_models:
        detail = "; ".join(f"{label}: {err}" for label, err in failures)
        raise ModelError(
            f"получено {len(rows)} прогнозов из {len(models)}, нужно минимум {min_models}. "
            f"Отказы — {detail}")
    if failures:
        log.warning("Продолжаю без %s (MIN_MODELS=%d)",
                    ", ".join(label for label, _ in failures), min_models)
    return rows
