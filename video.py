"""Видео «счёт появляется на табло» (first-frame + last-frame).

Почему два кадра, а не один: text-to-video / image-to-video модель сама
не напишет нужные цифры — она выдумает свои. Поэтому мы отдаём ей
первый кадр (пустой бланк) и последний (бланк с нашими цифрами), а модель
придумывает только то, КАК одно превращается в другое. Итоговые цифры
гарантированно совпадают с прогнозами и подписью поста.

Как именно — задаёт VIDEO_STYLE:
  cyber  (по умолчанию) — рук в кадре нет: цифры сами проявляются в клетках,
           как на крипто-терминале (барабан цифр → щелчок → свечение).
           Рука с маркером была самой хрупкой частью генерации: лишние
           пальцы, размазанные следы чернил, маркер на пол-кадра;
  marker — прежний вариант «рука вписывает счёт», оставлен как запасной.

Провайдеры (VIDEO_PROVIDER) — два разных API, оба поддерживают first-frame +
last-frame (нужно и там, и там: см. выше почему):

  Через OpenRouter (openrouter.ai/docs/guides/overview/multimodal/video-generation),
  тот же OPENROUTER_API_KEY, что уже используется для прогнозов в predictions.py:
    or-seedance-fast  (по умолчанию) — bytedance/seedance-2.0-fast, ~$0.04/сек.
                       Раньше по умолчанию стоял or-veo31lite — самый дешёвый
                       вариант, но реальные генерации показали, что он плохо
                       держит мелкий текст и рамки: текст плывёт уже на первом
                       кадре, проценты расползаются по всему экрану, клетка со
                       счётом иногда превращается в сплошную заливку без цифры.
                       Сменили на seedance-fast — сопоставимо по цене, качество
                       на этом сценарии ещё предстоит проверить;
    or-veo31lite      — google/veo-3.1-lite, $0.03/сек без звука на 720p — на
                       8-секундный сегмент это ~$0.24, тот же "почерк" Google
                       Veo, что и дорогой veo31, но кратно дешевле и, на
                       практике, кратно менее точный с мелким текстом;
    or-seedance-mini  — bytedance/seedance-2.0-mini, ~$0.034/сек, ещё дешевле,
                       но модель меньше — качество может просесть.

  Через fal.ai (нужен отдельный FAL_KEY):
    kling25 — fal-ai/kling-video/v2.5-turbo/pro/image-to-video: ~$0.70 за
              10-секундный сегмент, без звука;
    veo31   — fal-ai/veo3.1/first-last-frame-to-video: самый фотореалистичный,
              8 сек, 9:16, со звуком, но $0.40/сек — на SEGMENTS=2 это ~$6.4.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import random
import shutil
import subprocess
import time

import requests
from PIL import Image

from config import env_bool, env_float, env_int, env_str, require_env

log = logging.getLogger("video")

QUEUE = "https://queue.fal.run"
OR_API = "https://openrouter.ai/api/v1"

# fal.ai — нужен FAL_KEY
PROVIDERS = {
    "veo31": "fal-ai/veo3.1/first-last-frame-to-video",
    "kling25": "fal-ai/kling-video/v2.5-turbo/pro/image-to-video",
}

# OpenRouter — нужен OPENROUTER_API_KEY (тот же, что и для прогнозов)
OPENROUTER_MODELS = {
    "or-veo31lite": "google/veo-3.1-lite",
    "or-seedance-fast": "bytedance/seedance-2.0-fast",
    "or-seedance-mini": "bytedance/seedance-2.0-mini",
}

# fal у Kling режет prompt/negative_prompt на 2500 символов — держим с запасом,
# особенно NEGATIVE (он не зависит от числа строк, а prompt растёт с ними).

# VIDEO_STYLE=cyber (по умолчанию): рук в кадре нет вообще. Рука с маркером —
# самая хрупкая часть генерации: шесть пальцев, размазанные следы чернил,
# «возит маркером», перекрывает пол-постера. Цифрам, которые проявляются на
# табло сами, всё это просто негде сломаться.
NEGATIVE_CYBER = (
    "hands, fingers, arms, people, pen, pencil, marker, brush, ink, ink smears, smudges, "
    "streaks, handwriting, anything physically touching the device, device being picked up, "
    "device moving or tilting, phone-like rounded body, screen turning off, notification "
    "banners, status bar changes, "
    "camera movement, zoom, pan, "
    "shake, text changes, distorted letters, extra boxes, layout changes, moving board, "
    "watermark, ghost or duplicate digits, digits appearing before their turn, two boxes "
    "filling at once, digits in the wrong box, digits overflowing box edges, oversized digits, "
    "warped or melting numerals, unreadable numerals, flicker across the whole frame, "
    "idle pauses, dead time, action stopping before the video ends, random props, objects "
    "appearing or vanishing, background clutter not present at the start, "
    "already-filled box blanking out and refilling, digit disappearing then reappearing, "
    "loading or scramble animation repeating on a box that already shows its digit, finished "
    "row resetting to empty, VS badge turning "
    "into a number or percentage, progress ring or loading spinner replacing the VS badge, "
    "second percentage indicator, percentage or number appearing anywhere except next to "
    "AI ANALYSIS, glow or bloom effect on a row that is not currently being animated, glowing "
    "orb or blob artifacts, floating badge, distorted clock or status bar icons, animated fan "
    "spinning, moving parts on the device body, rotary dial turning, buttons lighting up or "
    "being pressed, core window flickering or changing pattern, "
    "sparks, spark burst, particle burst, particle explosion, confetti, fireworks, starburst, "
    "magical light burst, light rays radiating outward, glowing dust or embers, lens flare, "
    "number or percentage appearing in the title, number or percentage in the status bar or clock, "
    "number or percentage inside a model icon, model icon turning into a number or ring, digits or "
    "logo appearing on a plain crest placeholder card, crest placeholder card changing between "
    "frames, third percentage indicator, confidence number appearing early or flickering, "
    "wifi or signal bars, network or cloud icons, cables or charging cords")

# VIDEO_STYLE=marker — прежняя «рука вписывает счёт» (оставлена как запасная).
NEGATIVE_MARKER = (
    "text changes, distorted letters, extra boxes, moving paper, camera movement, zoom, "
    "blur, extra fingers, deformed hands, watermark, hand touching multiple boxes at once, "
    "fingers pointing at boxes, ghost or duplicate digits, digits appearing before written, "
    "wrong-box ink, two boxes filled at once, digits bleeding between rows, black/dark ink, "
    "ink color not matching marker tip, hand freezing mid-action, idle pauses, marker "
    "hovering without touching paper, dead time, action stopping before video ends, second "
    "hand, another hand holding the paper, digit popping in without the marker drawing it, "
    "ink appearing where the tip never travelled, instant or teleporting digits, extreme "
    "close-up, macro shot, hand or marker filling more than half the frame, oversized or "
    "bold digits, digits too big for the box, digits overflowing box edges, random props, "
    "decorative objects appearing or vanishing between shots, background clutter not present "
    "at the start")


def style() -> str:
    s = env_str("VIDEO_STYLE", "cyber").lower()
    return s if s in ("cyber", "marker") else "cyber"


def negative() -> str:
    return NEGATIVE_MARKER if style() == "marker" else NEGATIVE_CYBER


class VideoError(RuntimeError):
    pass


# ------------------------------------------------------------- требования ---

def ensure_tools() -> None:
    """Без ffmpeg склейка падает с FileNotFoundError из глубины subprocess —
    проверяем заранее и один раз, с внятным текстом."""
    missing = [t for t in ("ffmpeg", "ffprobe") if not shutil.which(t)]
    if missing:
        raise VideoError(
            f"Не найдены {', '.join(missing)} — установи ffmpeg "
            "(в Docker-образе это уже сделано; локально: apt install ffmpeg / brew install ffmpeg) "
            "или запусти с --no-video.")


# ---------------------------------------------------------------- промпт ---

def build_prompt(rows: list, first_row: int, last_row: int) -> str:
    """Промпт сегмента под текущий VIDEO_STYLE.

    Держим итоговую строку заметно короче 2500 символов (лимит fal/Kling на
    поле prompt) — она растёт с числом строк в сегменте, так что текст вокруг
    списка должен быть компактным, а не только сам список.
    """
    if style() == "marker":
        return _prompt_marker(rows, first_row, last_row)
    return _prompt_cyber(rows, first_row, last_row)


def _prompt_cyber(rows: list, first_row: int, last_row: int) -> str:
    lines = []
    for i, r in enumerate(rows[first_row:last_row], start=1):
        lines.append(f'{i}. "{r["label"].upper()}" row: "{r["home"]}" locks into its left box, '
                     f'then "{r["away"]}" locks into its right box.')
    order = "\n".join(lines)
    already = "Rows above are locked and finished — static, never flicker or scramble again.\n" if first_row else ""
    return (
        "Cinematic locked-off shot of a dedicated black offline AI terminal — a slim metal box with "
        "top vents, a status LED, a small lit core window, and a bottom panel of small buttons plus "
        "one large rotary dial; not a phone. It lies on a dark luxury desk, screen ON, showing the "
        "\"COINPLAY AI LAB\" app. The device never moves; NO hands, people, pens or markers ever "
        "enter frame — this is a screen recording of the device computing on its own.\n"
        "Mid-analysis, finishing during the clip: the \"AI ANALYSIS\" bar grows left to right with a "
        "green glow, its percentage counting up. The round VS badge always shows plain \"VS\", never "
        "a number. Each row's small icon glyph is a fixed logo, never a number or ring. Empty score "
        "boxes fill THEMSELVES, one row at a time, in this order:\n"
        f"{already}{order}\n"
        "Each digit: its box lights up, numerals scramble like an odds ticker for a beat, then snap "
        "into a glowing green digit with a scanline pulse and a glow contained to that one row only, "
        "never spreading to another row or a flag card. Only the active box glows; later boxes stay "
        "empty, earlier ones stay finished — a filled box never blanks out, flickers or replays its "
        "reveal again. Never two boxes, or two rows, animate at once.\n"
        "Digits appear ONLY in the two boxes named above. Flag cards, including any plain "
        "initials-only placeholder crest, are static photos — no digit, glow or logo ever appears or "
        "changes on them. The title, the clock and the team names never show a digit or percent sign.\n"
        "Right at the very end, once every row is filled, the AI CONSENSUS plate lights up: its "
        "confidence percentage brightens once beside the pick text, then holds steady — the only "
        "other number on screen besides AI ANALYSIS.\n"
        "Everything else — device body, vents, status LED, core window, control-panel buttons, "
        "rotary dial, serial plate, title, crests, VS badge, names, icons, layout, desk — stays "
        "perfectly still: the buttons never light up or get pressed, the dial never turns. "
        "Continuous rhythm, no idle pause. Rich blacks, green neon light, deep violet accents."
    )


def _prompt_marker(rows: list, first_row: int, last_row: int) -> str:
    lines = []
    for i, r in enumerate(rows[first_row:last_row], start=1):
        lines.append(f'{i}. "{r["label"].upper()}": write "{r["home"]}" in its left box, then '
                     f'"{r["away"]}" in its right box, same row.')
    order = "\n".join(lines)
    already = ("Rows above stay already-filled and untouched.\n") if first_row else ""
    return (
        "Locked-off smartphone shot from slightly above and to the side, zoomed OUT to a comfortable "
        "working distance — the whole printed \"COINPLAY AI LAB\" sheet fits inside the frame with "
        "visible plain wooden tabletop margin on all sides. Never a tight close-up; hand and marker "
        "stay modestly sized against the sheet, not filling half the frame. Sheet lies flat and "
        "still, weighted down on its own, no hand holding it.\n"
        "Only ONE hand is ever visible in the whole clip — holding a bright yellow/gold marker at a "
        "normal size relative to the sheet; ink is the SAME bright yellow as the marker (never "
        "black/dark). No second hand appears. The marker fills ONE empty box at a time, in this "
        "exact order:\n"
        f"{already}{order}\n"
        "Rules: the marker tip must be seen physically touching and dragging across the paper for "
        "every single digit — a digit only exists where the tip visibly travelled; it never pops in, "
        "appears instantly, or shows up on a box the tip hasn't reached. The tip touches at most one "
        "box at a time — the one being written — never brushing others. No two boxes fill "
        "simultaneously. Later rows stay blank while an earlier row is still being written, even "
        "though the end frame already shows them filled. Each digit: small, neat, precisely centered "
        "and sized to fit cleanly inside its box with margin around it — never large, bold, or "
        "sprawling past the box edges — drawn as one confident continuous yellow stroke, ink exactly "
        "where the tip drags. The hand moves smoothly and unhurriedly box to box "
        "with NO idle pause or freezing — visible drawing motion fills the entire clip, finishing "
        "exactly when done. Everything else (title, flags, icons, unreached boxes) stays sharp and "
        "unchanged. Soft daylight, realistic hand, subtle marker squeak. At the end the hand lifts "
        "out of frame."
    )


# ------------------------------------------------------------------ fal ----

def _data_uri(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _headers() -> dict:
    key = require_env("FAL_KEY", "Ключ fal.ai нужен для рендера видео. "
                                 "Без него запускай с --no-video.")
    return {"Authorization": f"Key {key}", "Content-Type": "application/json"}


def _get_json(url: str, timeout: int) -> dict:
    r = requests.get(url, headers=_headers(), timeout=timeout)
    if r.status_code >= 400:
        raise VideoError(f"fal {url} -> {r.status_code}: {r.text[:300]}")
    try:
        return r.json()
    except ValueError as e:
        raise VideoError(f"fal {url}: ответ не JSON: {r.text[:200]}") from e


def _run(endpoint: str, payload: dict, timeout: int | None = None) -> dict:
    timeout = timeout or env_int("FAL_TIMEOUT_SEC", 900, lo=60, hi=3600)
    r = requests.post(f"{QUEUE}/{endpoint}", headers=_headers(), json=payload, timeout=120)
    if r.status_code >= 400:
        raise VideoError(f"fal submit {endpoint} -> {r.status_code}: {r.text[:500]}")
    try:
        job = r.json()
    except ValueError as e:
        raise VideoError(f"fal submit {endpoint}: ответ не JSON: {r.text[:200]}") from e

    request_id = job.get("request_id")
    status_url = job.get("status_url")
    response_url = job.get("response_url")
    if not (status_url and response_url):
        if not request_id:
            raise VideoError(f"fal: в ответе нет ни request_id, ни ссылок: {str(job)[:300]}")
        status_url = status_url or f"{QUEUE}/{endpoint}/requests/{request_id}/status"
        response_url = response_url or f"{QUEUE}/{endpoint}/requests/{request_id}"
    log.info("fal: задача %s поставлена (%s)", request_id, endpoint)

    deadline = time.time() + timeout
    delay, last_status = 3.0, ""
    while time.time() < deadline:
        time.sleep(delay)
        delay = min(delay * 1.3, 15.0)  # мягкий backoff: не долбим статус каждые 6 сек час подряд
        try:
            s = _get_json(status_url, timeout=60)
        except (VideoError, requests.RequestException) as e:
            log.warning("fal: статус недоступен (%s) — повторю", e)
            continue
        st = (s.get("status") or "").upper()
        if st != last_status:
            log.info("fal: %s", st or "?")
            last_status = st
        if st == "COMPLETED":
            break
        if st in ("FAILED", "ERROR", "CANCELLED"):
            raise VideoError(f"fal: задача упала: {str(s)[:400]}")
    else:
        raise VideoError(f"fal: не дождался результата за {timeout} сек (request_id={request_id})")

    return _get_json(response_url, timeout=120)


def _video_url(result: dict) -> str:
    """У разных эндпоинтов fal результат лежит то в `video`, то в `videos[0]`."""
    node = result.get("video")
    if isinstance(node, dict) and node.get("url"):
        return node["url"]
    if isinstance(node, str) and node:
        return node
    for item in result.get("videos") or []:
        if isinstance(item, dict) and item.get("url"):
            return item["url"]
        if isinstance(item, str) and item:
            return item
    raise VideoError(f"fal: в ответе нет видео: {str(result)[:300]}")


def _download(url: str, out_path: str, headers: dict | None = None) -> None:
    tmp = f"{out_path}.part"
    with requests.get(url, headers=headers, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                if chunk:
                    f.write(chunk)
    if os.path.getsize(tmp) < 10_000:  # пустышка вместо ролика — лучше узнать сразу
        os.remove(tmp)
        raise VideoError(f"скачанный файл подозрительно мал ({url})")
    os.replace(tmp, out_path)


# ------------------------------------------------------------ openrouter ---

def _or_headers() -> dict:
    key = require_env(
        "OPENROUTER_API_KEY",
        "Ключ OpenRouter нужен и для видео (or-*), и для прогнозов моделей — "
        "один и тот же ключ.")
    return {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": env_str("OPENROUTER_REFERER", "https://t.me/aimatchlab"),
        "X-Title": "AI Match Lab",
    }


def _or_frame(img: Image.Image, frame_type: str) -> dict:
    return {"type": "image_url", "image_url": {"url": _data_uri(img)}, "frame_type": frame_type}


def _or_submit(model: str, prompt: str, first: Image.Image, last: Image.Image) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        # По умолчанию SEGMENTS даёт 1 строку (2 клетки) на сегмент — 4 сек с
        # запасом хватает руке физически дойти и коснуться обеих клеток; для
        # veo-3.1-lite это ещё и минимально короткая из поддерживаемых (4/6/8).
        "duration": env_int("OR_VIDEO_DURATION", 4, lo=1, hi=15),
        "resolution": env_str("OR_VIDEO_RESOLUTION", "720p"),
        "aspect_ratio": "9:16",
        "generate_audio": env_bool("OR_VIDEO_AUDIO", False),
        "frame_images": [_or_frame(first, "first_frame"), _or_frame(last, "last_frame")],
    }
    r = requests.post(f"{OR_API}/videos", headers=_or_headers(), json=payload, timeout=120)
    if r.status_code >= 400:
        raise VideoError(f"openrouter submit {model} -> {r.status_code}: {r.text[:500]}")
    try:
        job = r.json()
    except ValueError as e:
        raise VideoError(f"openrouter submit {model}: ответ не JSON: {r.text[:200]}") from e
    job_id = job.get("id")
    if not job_id:
        raise VideoError(f"openrouter: в ответе нет id задачи: {str(job)[:300]}")
    log.info("openrouter: задача %s поставлена (%s)", job_id, model)
    return job_id


def _or_poll(job_id: str, timeout: int | None = None) -> dict:
    timeout = timeout or env_int("OR_VIDEO_TIMEOUT_SEC", 900, lo=60, hi=3600)
    url = f"{OR_API}/videos/{job_id}"
    deadline = time.time() + timeout
    delay, last_status = 3.0, ""
    while time.time() < deadline:
        time.sleep(delay)
        delay = min(delay * 1.3, 15.0)  # мягкий backoff, как у fal-луп ниже
        try:
            r = requests.get(url, headers=_or_headers(), timeout=60)
            r.raise_for_status()
            s = r.json()
        except (requests.RequestException, ValueError) as e:
            log.warning("openrouter: статус недоступен (%s) — повторю", e)
            continue
        st = (s.get("status") or "").lower()
        if st != last_status:
            log.info("openrouter: %s", st or "?")
            last_status = st
        if st == "completed":
            cost = (s.get("usage") or {}).get("cost")
            if cost is not None:
                log.info("openrouter: сегмент стоил $%s", cost)
            return s
        if st in ("failed", "cancelled", "expired"):
            raise VideoError(f"openrouter: задача {st}: {str(s)[:400]}")
    raise VideoError(f"openrouter: не дождался результата за {timeout} сек (id={job_id})")


def _or_video_url(result: dict) -> str:
    urls = result.get("unsigned_urls") or []
    if urls and urls[0]:
        return urls[0]
    raise VideoError(f"openrouter: в ответе нет unsigned_urls: {str(result)[:300]}")


def _payload(provider: str, first: Image.Image, last: Image.Image, prompt: str) -> dict:
    if provider == "veo31":
        return {
            "prompt": prompt,
            "first_frame_url": _data_uri(first),
            "last_frame_url": _data_uri(last),
            "duration": env_str("VEO_DURATION", "8s"),
            "aspect_ratio": "9:16",
            "resolution": env_str("VEO_RESOLUTION", "1080p"),
            "generate_audio": env_bool("VEO_AUDIO", True),
            "negative_prompt": negative(),
        }
    return {
        "prompt": prompt,
        "image_url": _data_uri(first),
        "tail_image_url": _data_uri(last),
        "duration": env_str("KLING_DURATION", "10"),
        "negative_prompt": negative(),
        # Пробовали поднять до 0.8 — стало хуже: модель агрессивнее "подгоняет"
        # кадры под последний референс (уже полностью заполненный бланк) и
        # цифры начинают появляться в клетках РАНЬШЕ, чем маркер до них
        # долистал ("прыгает по клеткам"), вместо честного покадрового письма.
        # Вернули дефолт на 0.6 — это, а не рост cfg, снижает "отсебятину".
        "cfg_scale": env_float("KLING_CFG", 0.6, lo=0.0, hi=1.0),
    }


def generate_segment(first: Image.Image, last: Image.Image, prompt: str, out_path: str) -> str:
    provider = env_str("VIDEO_PROVIDER", "or-seedance-fast").lower()
    if provider in OPENROUTER_MODELS:
        model = OPENROUTER_MODELS[provider]

        def _once():
            job_id = _or_submit(model, prompt, first, last)
            result = _or_poll(job_id)
            # "unsigned_urls" — обманчивое название: это не публичная presigned-
            # ссылка (как у fal), а собственный content-эндпоинт OpenRouter,
            # ему всё равно нужен тот же Bearer-токен, что и на submit/poll —
            # без заголовка отдаёт 401, и сегмент проваливается уже ПОСЛЕ того,
            # как OpenRouter списал деньги за генерацию.
            _download(_or_video_url(result), out_path, headers=_or_headers())
    elif provider in PROVIDERS:
        endpoint = PROVIDERS[provider]
        payload = _payload(provider, first, last, prompt)

        def _once():
            result = _run(endpoint, payload)
            _download(_video_url(result), out_path)
    else:
        known = sorted(OPENROUTER_MODELS) + sorted(PROVIDERS)
        raise VideoError(f"VIDEO_PROVIDER={provider!r} — известны только {', '.join(known)}")

    attempts = env_int("FAL_ATTEMPTS", 2, lo=1, hi=5)
    last_err: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            _once()
            log.info("Сегмент сохранён: %s", out_path)
            return out_path
        except (VideoError, requests.RequestException) as e:
            last_err = e
            log.warning("видео: попытка %d/%d не удалась — %s", attempt, attempts, e)
            if attempt < attempts:
                time.sleep(5 * attempt + random.uniform(0, 2))
    raise VideoError(f"Не удалось сгенерировать сегмент за {attempts} попыт(ки): {last_err}")


# ------------------------------------------------------------- ffmpeg ------

def _ff(*args) -> None:
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *args]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        # раньше здесь был check=True, и настоящая причина (сообщение ffmpeg)
        # просто терялась — в логе оставался только код возврата
        raise VideoError(f"ffmpeg завершился с кодом {proc.returncode}:\n"
                         f"{(proc.stderr or '').strip()[:800]}")


def _has_audio(path: str) -> bool:
    proc = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a",
                           "-show_entries", "stream=index", "-of", "csv=p=0", path],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        log.warning("ffprobe не смог прочитать %s — считаю, что звука нет", path)
        return False
    return bool(proc.stdout.strip())


def _normalize(src: str, dst: str) -> None:
    """Один формат для склейки: 1080x1920, 30 fps, H.264 + AAC (тишина, если звука нет)."""
    vf = "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,fps=30,format=yuv420p"
    common = ["-c:v", "libx264", "-preset", "medium", "-crf", "19",
              "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-ac", "2",
              "-video_track_timescale", "90000"]  # одинаковый timebase — иначе concat рассинхронит звук
    if _has_audio(src):
        _ff("-i", src, "-vf", vf, "-map", "0:v:0", "-map", "0:a:0", *common, dst)
    else:
        _ff("-i", src, "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo",
            "-vf", vf, "-map", "0:v:0", "-map", "1:a:0", "-shortest", *common, dst)


def assemble(segments: list, out_path: str, hold_sec: float = 2.0, music: str = "") -> str:
    """Склейка сегментов + стоп-кадр в конце, чтобы прогнозы успели прочитать."""
    ensure_tools()
    segments = [s for s in segments if s and os.path.exists(s) and os.path.getsize(s) > 0]
    if not segments:
        raise VideoError("Нечего склеивать: ни одного готового сегмента")

    work = os.path.dirname(os.path.abspath(out_path)) or "."
    os.makedirs(work, exist_ok=True)
    temp: list[str] = []
    try:
        norm = []
        for i, s in enumerate(segments):
            n = os.path.join(work, f"_norm{i}.mp4")
            _normalize(s, n)
            norm.append(n)
        temp += norm

        if len(norm) == 1:
            joined = norm[0]
        else:
            lst = os.path.join(work, "_concat.txt")
            with open(lst, "w", encoding="utf-8") as f:
                for n in norm:
                    # в concat-листе кавычка внутри пути экранируется как '\''
                    safe = os.path.abspath(n).replace("'", r"'\''")
                    f.write(f"file '{safe}'\n")
            joined = os.path.join(work, "_joined.mp4")
            temp += [lst, joined]
            _ff("-f", "concat", "-safe", "0", "-i", lst, "-fflags", "+genpts", "-c", "copy", joined)

        hold_sec = max(0.0, hold_sec)
        if music and os.path.exists(music):
            # MUSIC_OFFSET — с какой секунды трека начинать: так пик/дроп
            # трека можно подвести ровно под момент раскрытия вердикта
            # (конец последних сегментов + начало hold-кадра), а не всегда
            # слушать начало файла. atrim идёт ПОСЛЕ stream_loop, поэтому
            # смещение работает даже если сам трек короче видео.
            offset = env_float("MUSIC_OFFSET", 0.0, lo=0.0, hi=600.0)
            _ff("-i", joined, "-stream_loop", "-1", "-i", music,
                "-filter_complex",
                f"[0:v]tpad=stop_mode=clone:stop_duration={hold_sec}[v];"
                f"[0:a]apad=pad_dur={hold_sec}[a0];"
                f"[1:a]atrim=start={offset},asetpts=PTS-STARTPTS,"
                f"volume={env_float('MUSIC_VOLUME', 0.25, lo=0.0, hi=1.0)}[a1];"
                f"[a0][a1]amix=inputs=2:duration=first:normalize=0[a]",
                "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-preset", "medium",
                "-crf", "19", "-c:a", "aac", "-b:a", "160k",
                "-movflags", "+faststart", out_path)
        else:
            _ff("-i", joined, "-vf", f"tpad=stop_mode=clone:stop_duration={hold_sec}",
                "-af", f"apad=pad_dur={hold_sec}", "-c:v", "libx264", "-preset", "medium",
                "-crf", "19", "-c:a", "aac", "-b:a", "160k",
                "-movflags", "+faststart", out_path)
    finally:
        for p in temp:
            if p == out_path:
                continue
            try:
                os.remove(p)
            except OSError:
                pass
    return out_path
