from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from api.app import create_app
from tests.test_dispute_api import (
    _make_valid_pack_zip,
    _tamper_zip_file_and_update_manifest,
)


def test_http_dispute_verify_valid_pack_ok(tmp_path: Path):
    app = create_app()
    client = TestClient(app)

    pack = _make_valid_pack_zip(tmp_path)

    with open(pack, "rb") as f:
        resp = client.post("/dispute/verify", files={"file": ("pack.zip", f, "application/zip")})

    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert data["por_root"].startswith("0x")


def test_http_dispute_verify_tamper_evidence_kind(tmp_path: Path):
    app = create_app()
    client = TestClient(app)

    pack = _make_valid_pack_zip(tmp_path)
    tampered = tmp_path / "tamper_evidence.zip"

    def mutate(o: dict):
        o["items"][0]["extracted_fields"]["price_usd"] = 123

    _tamper_zip_file_and_update_manifest(
        src_zip=pack,
        dst_zip=tampered,
        inner_path="evidence_bundle.json",
        mutate_fn=mutate,
    )

    with open(tampered, "rb") as f:
        resp = client.post("/dispute/verify", files={"file": ("pack.zip", f, "application/zip")})

    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is False
    assert data["challenge_ref"]["kind"] == "evidence_leaf"


def test_http_dispute_verify_missing_file_400():
    app = create_app()
    client = TestClient(app)

    resp = client.post("/dispute/verify")
    assert resp.status_code == 400
    data = resp.json()
    assert data["ok"] is False
    assert data["challenge_ref"]["kind"] == "pack_format"
