from __future__ import annotations

import json
from urllib.parse import urlparse
from typing import Any

from .base import SiteExtractor, ExtractionError


class HLTVPlayerWeaponStatsExtractor(SiteExtractor):
    """Extractor for HLTV player weapon stats pages.

    HLTV is usually Cloudflare-protected, so plain HTTP fetch often returns a
    challenge page. We therefore use Gemini UrlContext (if provided) to read
    the page and extract the AWP frags/kills count.
    """

    source_id = "hltv"

    def can_handle(self, url: str) -> bool:
        try:
            p = urlparse(url)
            host = (p.netloc or "").lower().lstrip("www.")
            path = (p.path or "")
            return host.endswith("hltv.org") and (
                "/stats/players/" in path or "/stats/players/weapon/" in path
            )
        except Exception:
            return False

    def extract_and_summarize(
        self,
        url: str,
        http_client: Any,
        *,
        gemini_client: Any | None = None,
        gemini_model: str | None = None,
    ) -> tuple[str, dict[str, Any]]:
        if gemini_client is None or not gemini_model:
            raise ExtractionError("HLTV extractor requires gemini_client + gemini_model (UrlContext)")

        from google.genai import types

        base_instructions = (
            "You are extracting structured data from an HLTV stats page.\n"
            "Read the provided URL using UrlContext and extract the player's weapon statistics.\n\n"
            f"URL: {url}\n\n"
            "Return ONLY valid JSON (no markdown) with this schema:\n"
            "{\n"
            '  "awp_kills": <integer or null>,\n'
            '  "awp_row_quote": "verbatim short excerpt of the AWP row/line",\n'
            '  "notes": "where it was found (table/row/column)"\n'
            "}\n\n"
            "Rules:\n"
            "- You MUST locate any weapon breakdown section (table or list).\n"
            "- Find the weapon labeled 'AWP' (case-insensitive).\n"
            "- Extract the numeric count of kills/frags with the AWP (integer).\n"
            "- Include awp_row_quote as a verbatim excerpt that contains both 'AWP' and the number.\n"
            "- If the AWP value cannot be determined, set awp_kills to null and explain in notes.\n"
        )

        def _call(target_url: str, *, force_dump: bool = False) -> Any:
            p = base_instructions.replace(f"URL: {url}", f"URL: {target_url}")
            if force_dump:
                p += (
                    "\nIf you cannot find an AWP row, do this instead: return awp_kills=null and set awp_row_quote to the closest line you can find that mentions 'AWP'. "
                    "If the page contains NO mention of AWP at all, say that in notes."
                )
            return gemini_client.models.generate_content(
                model=gemini_model,
                contents=p,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    tools=[types.Tool(url_context=types.UrlContext())],
                ),
            )

        # Try the given URL first, then a weapon-history variant.
        candidate_urls = [url]
        if "/stats/players/" in url and "/stats/players/weapon/" not in url:
            candidate_urls.append(url.replace("/stats/players/", "/stats/players/weapon/"))

        last_raw = ""
        obj: dict[str, Any] | None = None
        for idx, cu in enumerate(candidate_urls):
            resp = _call(cu, force_dump=(idx > 0))

            # Extract text parts
            texts: list[str] = []
            for cand in getattr(resp, "candidates", []) or []:
                content = getattr(cand, "content", None)
                if not content:
                    continue
                for part in (getattr(content, "parts", None) or []):
                    t = getattr(part, "text", None)
                    if t:
                        texts.append(t)
            raw = "\n".join(texts).strip()
            last_raw = raw

            # Parse first JSON object
            try:
                dec = json.JSONDecoder()
                start = raw.find("{")
                if start >= 0:
                    parsed, _ = dec.raw_decode(raw[start:])
                else:
                    parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    obj = parsed
            except Exception:
                obj = None

            if obj is not None and obj.get("awp_kills") is not None:
                # success
                url = cu
                break

        if obj is None:
            raise ExtractionError(f"Gemini JSON parse failed; raw={last_raw[:200]}")

        awp_kills = obj.get("awp_kills")
        if awp_kills is not None:
            try:
                awp_kills = int(awp_kills)
            except Exception:
                awp_kills = None

        notes = str(obj.get("notes") or "")
        awp_row_quote = str(obj.get("awp_row_quote") or "")

        if awp_kills is None:
            summary = f"HLTV weapon stats: could not extract AWP kills/frags from {url}. Notes: {notes}"
        else:
            summary = f"HLTV weapon stats from {url}: AWP kills/frags = {awp_kills}. Notes: {notes}"

        meta = {
            "source_url": url,
            "awp_kills": awp_kills,
            "awp_row_quote": awp_row_quote,
        }
        return summary, meta
