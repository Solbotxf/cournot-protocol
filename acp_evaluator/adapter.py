"""
Adapter: Converts ACP Job deliverables into Cournot PromptSpec inputs
for verifiable evaluation.

Supports image deliverables via URL pre-validation.
"""
import json
import hashlib
import re
from datetime import datetime, timezone
from typing import Optional

from acp_evaluator.image_validator import check_image_url, build_image_evidence


# Patterns to detect image URLs in deliverables
IMAGE_URL_PATTERN = re.compile(
    r'https?://[^\s"\'<>]+\.(?:png|jpg|jpeg|gif|webp|svg|bmp|tiff)',
    re.IGNORECASE,
)

IMAGE_MARKDOWN_PATTERN = re.compile(
    r'!\[.*?\]\((https?://[^\s)]+)\)',
)


def extract_image_urls(deliverable: str) -> list[str]:
    """Extract image URLs from deliverable text."""
    urls = set()
    # Markdown images
    for match in IMAGE_MARKDOWN_PATTERN.finditer(deliverable):
        urls.add(match.group(1))
    # Direct URLs
    for match in IMAGE_URL_PATTERN.finditer(deliverable):
        urls.add(match.group(0))
    # JSON field
    try:
        data = json.loads(deliverable)
        if isinstance(data, dict):
            for key in ("url", "image_url", "value", "image", "src"):
                val = data.get(key, "")
                if isinstance(val, str) and val.startswith("http"):
                    urls.add(val)
    except (json.JSONDecodeError, TypeError):
        pass
    return list(urls)


def validate_image_deliverable(deliverable: str) -> Optional[dict]:
    """
    If deliverable contains image URLs, validate them and return evidence.
    Returns None if no image URLs found.
    """
    urls = extract_image_urls(deliverable)
    if not urls:
        return None

    results = []
    all_valid = True
    for url in urls:
        check = check_image_url(url)
        results.append(build_image_evidence(check))
        if not check.is_valid_image:
            all_valid = False

    return {
        "image_urls_found": len(urls),
        "all_valid": all_valid,
        "checks": results,
    }


def build_evaluation_query(
    job_description: str,
    deliverable: str,
    job_metadata: dict = None,
) -> str:
    """
    Build a natural-language query for Cournot's PoR pipeline.

    For image deliverables, pre-validates URLs and injects evidence.
    """
    # Pre-validate image URLs
    image_evidence = validate_image_deliverable(deliverable)

    evidence_block = ""
    if image_evidence:
        evidence_block = (
            "\n\n=== IMAGE URL VALIDATION (pre-collected evidence) ===\n"
            f"Image URLs found: {image_evidence['image_urls_found']}\n"
            f"All valid: {image_evidence['all_valid']}\n"
        )
        for check in image_evidence["checks"]:
            evidence_block += f"\n  URL: {check['url']}\n"
            evidence_block += f"  {check['evidence_summary']}\n"

    meta_str = ""
    if job_metadata:
        meta_str = f"\n\nAdditional job metadata:\n{json.dumps(job_metadata, indent=2)}"

    query = (
        f"Evaluate whether the following deliverable satisfies the job specification.\n\n"
        f"=== JOB SPECIFICATION ===\n{job_description}\n\n"
        f"=== DELIVERABLE ===\n{deliverable}\n"
        f"{evidence_block}"
        f"{meta_str}\n\n"
        f"Answer YES if the deliverable fully satisfies the specification. "
        f"Answer NO if it clearly does not. "
        f"Answer INVALID if the specification is ambiguous or there is insufficient information to judge."
    )
    return query


def build_evaluation_reason(
    verdict: str,
    confidence: float,
    reasoning_summary: str,
    por_root: str,
) -> str:
    """Build the reason string for ACP's complete()/reject()."""
    result = {
        "evaluator": "cournot-protocol",
        "verdict": verdict,
        "confidence": confidence,
        "reasoning_summary": reasoning_summary[:500],
        "por_root": por_root,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }
    return json.dumps(result)


def deliverable_content_hash(deliverable: str) -> str:
    """SHA-256 hash of deliverable content."""
    return hashlib.sha256(deliverable.encode("utf-8")).hexdigest()
