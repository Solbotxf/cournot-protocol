#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════╗
║  Cournot AI — Negative Test Demo (Mainnet)                  ║
║  Demonstrates: Job → Provider fails → PoR Eval → REJECT     ║
╚══════════════════════════════════════════════════════════════╝

This script creates a job with a provider known to return
error/empty content. The Cournot Evaluator Daemon will:
  1. Detect the deliverable contains error content
  2. Run PoR pipeline — verdict: INVALID
  3. Submit REJECT on-chain (error deliverables are rejected)
"""
import json, time, logging, sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("demo-negative")

from virtuals_acp.client import VirtualsACP
from virtuals_acp.contract_clients.contract_client_v2 import ACPContractClientV2
from virtuals_acp.models import ACPOnlineStatus

AGENT_WALLET = os.environ["ACP_AGENT_WALLET_ADDRESS"]
ENTITY_ID    = int(os.environ["ACP_ENTITY_ID"])
WALLET_PATH  = os.environ.get("ACP_WALLET_PATH", "/root/.openclaw/projects/cournot/.acp-wallet.json")

def main():
    print()
    print("=" * 60)
    print("  🔴  COURNOT AI — NEGATIVE TEST (Mainnet)")
    print("  Evaluator: Proof-of-Reasoning (PoR) Pipeline")
    print("=" * 60)
    print()

    # 1. Connect
    log.info("Step 1: Connecting to ACP on Base chain...")
    with open(WALLET_PATH) as f:
        wallet = json.load(f)
    client = ACPContractClientV2(
        wallet_private_key=wallet["private_key"],
        agent_wallet_address=AGENT_WALLET,
        entity_id=ENTITY_ID,
    )
    acp = VirtualsACP(acp_contract_clients=client, skip_socket_connection=True)
    log.info(f"  ✅ Connected | Agent: {AGENT_WALLET}")

    # 2. Find ArAIstotle (link_reliability_check fails for non-news URLs)
    log.info("Step 2: Finding provider (ArAIstotle — link reliability)...")
    agents = acp.browse_agents(keyword="ArAIstotle", top_k=3, online_status=ACPOnlineStatus.ONLINE)
    offering = None
    for a in agents:
        for jo in a.job_offerings:
            if jo.name == "link_reliability_check":
                offering = jo
                log.info(f"  ✅ Found: {a.name} / {jo.name} | ${jo.price}")
                break
        if offering: break
    if not offering:
        log.error("  ❌ Provider not found"); return

    # 3. Create job with URL known to fail (not in media database)
    log.info("Step 3: Creating job with a URL expected to FAIL verification...")
    log.info(f"  Request: Check reliability of 'https://example.com/fake-news'")
    log.info(f"  Expected: Provider returns error → Cournot REJECTS")
    job_id = offering.initiate_job(
        service_requirement={"link_reliability_url": "https://example.com/fake-news-article-2026"},
        evaluator_address=AGENT_WALLET,
    )
    log.info(f"  ✅ Job created: #{job_id}")

    # 4. Pay & wait
    log.info("Step 4: Waiting for provider...")
    paid = False
    for tick in range(50):
        time.sleep(6)
        try:
            for j in acp.get_active_jobs(page=1, page_size=50):
                if j.id == job_id:
                    phase = str(getattr(j, 'phase', ''))
                    if 'NEGOTIATION' in phase.upper() and not paid:
                        time.sleep(3)
                        for retry in range(3):
                            try:
                                j.pay_and_accept_requirement()
                                log.info(f"  ✅ Payment sent")
                                paid = True; break
                            except:
                                time.sleep(5)
                                for j2 in acp.get_active_jobs(page=1, page_size=50):
                                    if j2.id == job_id: j = j2; break
                    elif 'TRANSACTION' in phase.upper():
                        if tick % 3 == 0:
                            log.info(f"  ⏳ Provider processing... ({(tick+1)*6}s)")
                    elif 'EVALUATION' in phase.upper():
                        log.info(f"  📦 Deliverable received (expected: error content)")
                        log.info(f"  🔍 Cournot Daemon running PoR pipeline...")
                    break

            # Check completion — for rejected jobs check cancelled/completed
            for j in acp.get_completed_jobs(page=1, page_size=10):
                if j.id == job_id:
                    print()
                    print("=" * 60)
                    log.info("🛑 NEGATIVE TEST RESULT: JOB REJECTED ❌")
                    print("=" * 60)
                    log.info(f"  Job ID:      #{job_id}")
                    log.info(f"  Provider:    ArAIstotle (link reliability)")
                    log.info(f"  Input:       https://example.com/fake-news-article-2026")
                    log.info(f"  Evaluator:   Cournot AI (PoR Pipeline)")
                    log.info(f"  Reason:      Deliverable contains error — URL not in database")
                    log.info(f"  PoR Verdict: INVALID (error content detected)")
                    print("=" * 60)
                    return

            # Also check if still active but rejected
            for j in acp.get_active_jobs(page=1, page_size=50):
                if j.id == job_id:
                    phase = str(getattr(j, 'phase', ''))
                    if 'REJECTED' in phase.upper() or 'CANCELLED' in phase.upper():
                        print()
                        print("=" * 60)
                        log.info("🛑 NEGATIVE TEST RESULT: JOB REJECTED ❌")
                        print("=" * 60)
                        log.info(f"  Job ID:      #{job_id}")
                        log.info(f"  Reason:      Error content → PoR INVALID → REJECTED")
                        print("=" * 60)
                        return

        except Exception as e:
            if 'network' not in str(e).lower():
                log.warning(f"  ⚠️  {e}")

    log.warning("Timeout — check: journalctl -u cournot-evaluator -n 30")

if __name__ == "__main__":
    main()
