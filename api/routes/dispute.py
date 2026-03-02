"""Dispute routes (Off-chain).

Provides HTTP endpoints for dispute/challenge verification.

- POST /dispute/verify (multipart/form-data)

HTTP layer does only I/O + parameter parsing; core logic is implemented in
`api.dispute_verify.verify_pack`.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, UploadFile
from starlette.responses import JSONResponse

from api.dispute_verify import verify_pack
from api.models.responses import DisputeVerifyResponse


router = APIRouter(tags=["dispute"], prefix="/dispute")


@router.post("/verify", response_model=DisputeVerifyResponse)
async def dispute_verify_http(
    file: UploadFile | None = File(default=None, description="Artifact pack zip file"),
    expected_por_root: str | None = Form(default=None, description="Optional expected por_root"),
    pack_uri: str | None = Form(default=None, description="Optional pack_uri override (e.g. ipfs://CID)"),
    baseline_file: UploadFile | None = File(default=None, description="Optional baseline pack zip"),
):
    """Verify an uploaded pack for dispute purposes.

    Returns:
      - 200 for both ok=true/false
      - 400 only when request is invalid (e.g. missing file)
    """

    if file is None or file.filename in (None, ""):
        body = {
            "ok": False,
            "por_root": None,
            "expected_por_root": expected_por_root,
            "pack_uri": pack_uri or "",
            "sentinel_version": "SentinelStrict:v1",
            "timestamp": None,
            "challenge_ref": {"kind": "pack_format", "leaf_index": None, "reason": "missing file"},
            "errors": ["missing file"],
            "checks": [],
        }
        return JSONResponse(status_code=400, content=body)

    tmp_pack: Path | None = None
    tmp_baseline: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            tmp.write(await file.read())
            tmp_pack = Path(tmp.name)

        if baseline_file is not None and baseline_file.filename not in (None, ""):
            with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmpb:
                tmpb.write(await baseline_file.read())
                tmp_baseline = Path(tmpb.name)

        report = verify_pack(
            tmp_pack,
            expected_por_root=expected_por_root,
            pack_uri=pack_uri or file.filename,
            baseline_pack=tmp_baseline,
        )
        return JSONResponse(status_code=200, content=report)

    finally:
        if tmp_pack is not None:
            tmp_pack.unlink(missing_ok=True)
        if tmp_baseline is not None:
            tmp_baseline.unlink(missing_ok=True)
