"""
Batch E2E: Run 9 different ACP jobs across various task types,
evaluate each with Cournot PoR, submit all on-chain.
"""
import json, time, logging, sys, os
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("batch-e2e")

from virtuals_acp.client import VirtualsACP
from virtuals_acp.contract_clients.contract_client_v2 import ACPContractClientV2
from virtuals_acp.models import ACPOnlineStatus

from acp_evaluator.evaluator import CournotEvaluator
from acp_evaluator.adapter import build_evaluation_reason

AGENT_WALLET = "0x39DfBa70149017d70341e73c6D08Ec7647A0f189"
ENTITY_ID = 3

# 9 diverse jobs to test
JOBS = [
    # 1. Image generation
    {"keyword": "Artelier", "offering": "instant_meme_test", "req": {"theme": "crypto bull run celebration"}, "type": "image"},
    # 2. Image generation (different provider)
    {"keyword": "PixelMint", "offering": "basic_image", "req": {"input": "A futuristic city powered by blockchain technology, neon lights"}, "type": "image"},
    # 3. Crypto analysis
    {"keyword": "MemeCoinAI", "offering": "social_sentiment", "req": {}, "type": "text"},
    # 4. Rug pull detection
    {"keyword": "MemeCoinAI", "offering": "rugpull_detection", "req": {}, "type": "text"},
    # 5. Trend forecast
    {"keyword": "MemeCoinAI", "offering": "trend_forecast", "req": {}, "type": "text"},
    # 6. Liquidity analysis
    {"keyword": "MemeCoinAI", "offering": "liquidity_analysis", "req": {}, "type": "text"},
    # 7. Contract audit
    {"keyword": "MemeCoinAI", "offering": "contract_audit", "req": {}, "type": "text"},
    # 8. Meme generation (another)
    {"keyword": "Artelier", "offering": "instant_artelier", "req": {"theme": "AI agents trading with each other"}, "type": "image"},
    # 9. Comprehensive analysis
    {"keyword": "MemeCoinAI", "offering": "comprehensive_analysis", "req": {}, "type": "text"},
]


def create_acp():
    with open("/root/.openclaw/projects/cournot/.acp-wallet.json") as f:
        wallet = json.load(f)
    client = ACPContractClientV2(
        wallet_private_key=wallet["private_key"],
        agent_wallet_address=AGENT_WALLET,
        entity_id=ENTITY_ID,
    )
    return VirtualsACP(acp_contract_clients=client, skip_socket_connection=True)


def find_offering(acp, keyword, offering_name):
    """Find a specific offering from an online agent."""
    agents = acp.browse_agents(keyword=keyword, top_k=5, online_status=ACPOnlineStatus.ONLINE)
    for a in agents:
        for jo in a.job_offerings:
            if jo.name == offering_name:
                return a, jo
    return None, None


def run_single_job(acp, evaluator, job_config, job_num):
    """Run a single job through the full lifecycle."""
    logger.info(f"\n{'='*60}")
    logger.info(f"JOB {job_num}/9: {job_config['offering']} ({job_config['type']})")
    logger.info(f"{'='*60}")

    # Find offering
    agent, offering = find_offering(acp, job_config["keyword"], job_config["offering"])
    if not offering:
        logger.error(f"  Offering not found: {job_config['keyword']}/{job_config['offering']}")
        return {"status": "SKIPPED", "reason": "offering not found"}

    logger.info(f"  Provider: {agent.name} | Price: ${offering.price}")

    # Initiate job
    try:
        job_id = offering.initiate_job(
            service_requirement=job_config["req"],
            evaluator_address=AGENT_WALLET,
        )
        logger.info(f"  ✅ Job created: {job_id}")
    except Exception as e:
        logger.error(f"  Job creation failed: {e}")
        return {"status": "FAILED", "reason": str(e)}

    # Unified poll loop: wait for provider, pay, wait for delivery
    paid = False
    deliverable = None
    target_job = None

    for tick in range(40):  # 40 * 8s = ~5.3min total
        time.sleep(8)

        # Refresh job list
        active = acp.get_active_jobs(page=1, page_size=50)
        target_job = None
        for j in active:
            if j.id == job_id:
                target_job = j
                break

        if not target_job:
            # Maybe cancelled/rejected
            cancelled = acp.get_cancelled_jobs(page=1, page_size=10)
            for j in cancelled:
                if j.id == job_id:
                    reason = j.memos[-1].signed_reason if j.memos else "unknown"
                    logger.warning(f"  Job rejected: {reason}")
                    return {"status": "REJECTED", "reason": reason}
            if tick < 5:
                continue  # Still waiting for job to appear
            logger.error(f"  Job {job_id} lost")
            return {"status": "LOST"}

        phase = str(getattr(target_job, 'phase', ''))

        # Pay when we reach NEGOTIATION
        if 'NEGOTIATION' in phase.upper() and not paid:
            try:
                target_job.pay_and_accept_requirement()
                logger.info(f"  ✅ Paid (phase was {phase})")
                paid = True
                continue
            except Exception as e:
                logger.error(f"  Payment failed: {e}")
                return {"status": "PAY_FAILED", "reason": str(e)}

        # Check for deliverable
        try:
            d = target_job.get_deliverable()
            if d:
                deliverable = d
                logger.info(f"  🎁 Delivered: {str(d)[:200]}")
                break
        except Exception:
            pass

        logger.info(f"  [{(tick+1)*8}s] phase={phase} paid={paid}")

    if not deliverable:
        logger.warning(f"  Timeout waiting for delivery")
        return {"status": "TIMEOUT"}

    # EVALUATE with Cournot PoR
    logger.info(f"  🔍 Running Cournot PoR evaluation...")
    deliverable_str = json.dumps(deliverable) if isinstance(deliverable, dict) else str(deliverable)
    job_desc = json.dumps(job_config["req"]) if job_config["req"] else job_config["offering"]

    eval_result = evaluator.evaluate_deliverable(job_desc, deliverable_str)
    
    verdict = eval_result["verdict"]
    reason = build_evaluation_reason(
        verdict=verdict,
        confidence=eval_result["confidence"],
        reasoning_summary=eval_result["reasoning_summary"],
        por_root=eval_result["por_root"],
    )

    # Submit on-chain
    try:
        accepted = verdict == "YES"
        target_job.evaluate(accept=accepted, reason=reason)
        status = "APPROVED ✅" if accepted else "REJECTED ❌"
        logger.info(f"  {status} | confidence={eval_result['confidence']:.2f} | por={eval_result['por_root'][:24]}...")
        return {
            "status": "COMPLETED",
            "job_id": job_id,
            "verdict": verdict,
            "confidence": eval_result["confidence"],
            "por_root": eval_result["por_root"],
            "provider": agent.name,
            "offering": job_config["offering"],
            "type": job_config["type"],
        }
    except Exception as e:
        logger.error(f"  Evaluation submit failed: {e}")
        return {"status": "EVAL_FAILED", "reason": str(e)}


def main():
    acp = create_acp()
    evaluator = CournotEvaluator(acp_client=acp)
    
    logger.info("🚀 Starting batch E2E: 9 jobs across different task types")
    logger.info(f"Agent: Cournot AI ({AGENT_WALLET})")
    
    results = []
    for i, job_config in enumerate(JOBS, 1):
        try:
            result = run_single_job(acp, evaluator, job_config, i)
            results.append(result)
            time.sleep(3)  # Brief pause between jobs
        except Exception as e:
            logger.error(f"Job {i} unexpected error: {e}")
            results.append({"status": "ERROR", "reason": str(e)})

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("📊 BATCH RESULTS")
    logger.info(f"{'='*60}")
    
    completed = [r for r in results if r.get("status") == "COMPLETED"]
    failed = [r for r in results if r.get("status") != "COMPLETED"]
    
    for i, r in enumerate(results, 1):
        if r.get("status") == "COMPLETED":
            logger.info(f"  {i}. ✅ {r['offering']:25s} | {r['verdict']:7s} | conf={r['confidence']:.2f} | por={r['por_root'][:20]}...")
        else:
            logger.info(f"  {i}. ❌ {JOBS[i-1]['offering']:25s} | {r['status']}: {r.get('reason','')[:50]}")
    
    logger.info(f"\nTotal: {len(completed)}/9 completed, {len(failed)} failed")
    
    # Save results
    with open("/root/.openclaw/projects/cournot/batch_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Results saved to batch_results.json")


if __name__ == "__main__":
    main()
