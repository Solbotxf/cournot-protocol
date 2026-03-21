#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════╗
║  Cournot AI — Positive Test Demo (Mainnet)                  ║
║  Demonstrates: Job → Provider delivers → PoR Eval → APPROVE ║
╚══════════════════════════════════════════════════════════════╝

This script creates a real ACP job on Base chain.
The Cournot Evaluator Daemon (running 24/7) will automatically:
  1. Detect the job entering EVALUATION phase
  2. Run the 5-stage Proof-of-Reasoning pipeline
  3. Submit APPROVE with PoR Merkle root on-chain
"""
import json, time, logging, sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("demo-positive")

from virtuals_acp.client import VirtualsACP
from virtuals_acp.contract_clients.contract_client_v2 import ACPContractClientV2
from virtuals_acp.models import ACPOnlineStatus

# ── Config ──────────────────────────────────────────────────
AGENT_WALLET = os.environ["ACP_AGENT_WALLET_ADDRESS"]
ENTITY_ID    = int(os.environ["ACP_ENTITY_ID"])
WALLET_PATH  = os.environ.get("ACP_WALLET_PATH", "/root/.openclaw/projects/cournot/.acp-wallet.json")

def main():
    print()
    print("=" * 60)
    print("  🟢  COURNOT AI — POSITIVE TEST (Mainnet)")
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
    log.info(f"  ✅ Connected | Agent: {AGENT_WALLET} | Entity: {ENTITY_ID}")

    # 2. Find provider
    log.info("Step 2: Finding provider (Daredevil — NBA data)...")
    agents = acp.browse_agents(keyword="Daredevil", top_k=3, online_status=ACPOnlineStatus.ONLINE)
    offering = None
    for a in agents:
        for jo in a.job_offerings:
            if jo.name == "broadcast_nba":
                offering = jo
                log.info(f"  ✅ Found: {a.name} / {jo.name} | ${jo.price}")
                break
        if offering: break
    if not offering:
        log.error("  ❌ Provider not found"); return

    # 3. Create job
    log.info("Step 3: Creating job on-chain...")
    log.info(f"  Request: NBA standings for Boston Celtics")
    log.info(f"  Evaluator: Cournot AI ({AGENT_WALLET})")
    job_id = offering.initiate_job(
        service_requirement={"dataType": "standings", "teamName": "Celtics"},
        evaluator_address=AGENT_WALLET,
    )
    log.info(f"  ✅ Job created on Base chain: #{job_id}")

    # 4. Pay when provider responds
    log.info("Step 4: Waiting for provider negotiation...")
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
                                log.info(f"  ✅ Payment sent ($0.01 USDC)")
                                paid = True; break
                            except:
                                time.sleep(5)
                                for j2 in acp.get_active_jobs(page=1, page_size=50):
                                    if j2.id == job_id: j = j2; break
                    elif 'TRANSACTION' in phase.upper():
                        if tick % 3 == 0:
                            log.info(f"  ⏳ Provider processing... ({(tick+1)*6}s)")
                    elif 'EVALUATION' in phase.upper():
                        log.info(f"  📦 Deliverable received! Job now in EVALUATION phase")
                        log.info(f"  🔍 Cournot Daemon will now run PoR pipeline automatically...")
                        log.info(f"     Pipeline: PromptSpec → Collector → Auditor → Judge → Sentinel")
                    break

            # Check completion
            for j in acp.get_completed_jobs(page=1, page_size=10):
                if j.id == job_id:
                    print()
                    print("=" * 60)
                    log.info("🎉 POSITIVE TEST RESULT: JOB APPROVED ✅")
                    print("=" * 60)
                    log.info(f"  Job ID:      #{job_id}")
                    log.info(f"  Provider:    Daredevil (NBA data)")
                    log.info(f"  Evaluator:   Cournot AI (PoR Pipeline)")
                    log.info(f"  Chain:       Base (Coinbase L2)")
                    log.info(f"  Result:      APPROVED with Proof-of-Reasoning hash")
                    log.info(f"  View on ACP: https://app.virtuals.io/acp/agents/b2ocwooi56v2wec8ohkf1sej")
                    print("=" * 60)
                    return
        except Exception as e:
            if 'network' not in str(e).lower():
                log.warning(f"  ⚠️  {e}")

    log.warning("Timeout — provider may be slow. Check daemon logs: journalctl -u cournot-evaluator -n 30")

if __name__ == "__main__":
    main()
