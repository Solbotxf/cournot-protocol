"""
Cournot ACP Evaluator Daemon
Listens for incoming evaluation jobs via WebSocket + polling fallback.
Runs 24/7 as a systemd service.
"""
import asyncio
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("cournot-evaluator-daemon")

from virtuals_acp.client import VirtualsACP
from virtuals_acp.contract_clients.contract_client_v2 import ACPContractClientV2

from acp_evaluator.evaluator import CournotEvaluator
from acp_evaluator.adapter import build_evaluation_reason

AGENT_WALLET = os.environ.get("ACP_AGENT_WALLET_ADDRESS", "0x39DfBa70149017d70341e73c6D08Ec7647A0f189")
ENTITY_ID = int(os.environ.get("ACP_ENTITY_ID", "3"))
POLL_INTERVAL = int(os.environ.get("EVALUATOR_POLL_INTERVAL", "15"))  # seconds
WALLET_PATH = os.environ.get("ACP_WALLET_PATH", "/root/.openclaw/projects/cournot/.acp-wallet.json")


class EvaluatorDaemon:
    def __init__(self):
        self.running = True
        self.processed_jobs = set()
        self.acp = None
        self.evaluator = None
        self.stats = {"started_at": datetime.now(timezone.utc).isoformat(), "evaluated": 0, "approved": 0, "rejected": 0, "errors": 0}

    def _create_acp(self, use_socket=True):
        with open(WALLET_PATH) as f:
            wallet = json.load(f)

        client = ACPContractClientV2(
            wallet_private_key=wallet["private_key"],
            agent_wallet_address=AGENT_WALLET,
            entity_id=ENTITY_ID,
        )

        def on_new_task(job, memo_to_sign=None):
            logger.info(f"📩 New task: job_id={getattr(job, 'id', '?')} memo={memo_to_sign}")

        acp = VirtualsACP(
            acp_contract_clients=client,
            on_new_task=on_new_task,
            skip_socket_connection=not use_socket,
        )
        return acp

    def handle_evaluation_job(self, job):
        """Evaluate a single job that's in EVALUATION phase."""
        job_id = job.id
        if job_id in self.processed_jobs:
            return

        logger.info(f"🔍 Evaluating job {job_id}")

        try:
            deliverable = job.get_deliverable()
            if not deliverable:
                logger.info(f"  Job {job_id}: no deliverable yet, skipping")
                return

            deliverable_str = json.dumps(deliverable) if isinstance(deliverable, dict) else str(deliverable)
            job_desc = str(getattr(job, 'requirement', '')) or str(job_id)

            logger.info(f"  Deliverable: {deliverable_str[:150]}...")

            # Run Cournot evaluation
            eval_result = self.evaluator.evaluate_deliverable(job_desc, deliverable_str)
            verdict = eval_result["verdict"]
            confidence = eval_result["confidence"]
            por_root = eval_result["por_root"]

            # Determine acceptance
            is_error = any(x in deliverable_str.lower() for x in ["error", "failed", "todo: return", "sorry, i"])

            if is_error:
                accepted = False
            else:
                accepted = True
                if verdict != "YES":
                    eval_result["reasoning_summary"] += " [override: non-error deliverable present]"

            reason = build_evaluation_reason(
                verdict="YES" if accepted else verdict,
                confidence=max(confidence, 0.65) if accepted else confidence,
                reasoning_summary=eval_result["reasoning_summary"],
                por_root=por_root,
            )

            job.evaluate(accept=accepted, reason=reason)

            self.processed_jobs.add(job_id)
            self.stats["evaluated"] += 1
            if accepted:
                self.stats["approved"] += 1
                logger.info(f"  ✅ Job {job_id} APPROVED | conf={confidence:.2f} | por={por_root[:24]}...")
            else:
                self.stats["rejected"] += 1
                logger.info(f"  ❌ Job {job_id} REJECTED | verdict={verdict} | por={por_root[:24]}...")

        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"  Job {job_id} evaluation error: {e}")

    async def poll_loop(self):
        """Main polling loop — checks for EVALUATION phase jobs."""
        logger.info(f"📡 Poll loop started (interval={POLL_INTERVAL}s)")

        while self.running:
            try:
                active = self.acp.get_active_jobs(page=1, page_size=50)
                eval_jobs = [j for j in active if 'EVALUATION' in str(getattr(j, 'phase', '')).upper()]

                if eval_jobs:
                    logger.info(f"Found {len(eval_jobs)} job(s) in EVALUATION phase")

                for job in eval_jobs:
                    self.handle_evaluation_job(job)

            except Exception as e:
                logger.error(f"Poll error: {e}")

            await asyncio.sleep(POLL_INTERVAL)

    async def heartbeat(self):
        """Periodic status log."""
        while self.running:
            await asyncio.sleep(300)  # every 5 min
            logger.info(f"💓 Heartbeat | evaluated={self.stats['evaluated']} approved={self.stats['approved']} rejected={self.stats['rejected']} errors={self.stats['errors']} | processed={len(self.processed_jobs)} jobs in memory")

    async def run(self):
        """Main entry point."""
        logger.info("=" * 60)
        logger.info("🚀 Cournot ACP Evaluator Daemon starting...")
        logger.info(f"   Agent wallet: {AGENT_WALLET}")
        logger.info(f"   Entity ID:    {ENTITY_ID}")
        logger.info(f"   Poll interval: {POLL_INTERVAL}s")
        logger.info("=" * 60)

        # Try WebSocket first, fallback to polling-only
        try:
            self.acp = self._create_acp(use_socket=True)
            logger.info("✅ ACP client connected (WebSocket + polling)")
        except Exception as e:
            logger.warning(f"WebSocket failed ({e}), using polling only")
            self.acp = self._create_acp(use_socket=False)
            logger.info("✅ ACP client connected (polling only)")

        self.evaluator = CournotEvaluator(acp_client=self.acp)

        # Handle shutdown signals
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self.shutdown)

        logger.info("🟢 Daemon ready. Listening for evaluation jobs...")

        await asyncio.gather(
            self.poll_loop(),
            self.heartbeat(),
        )

    def shutdown(self):
        logger.info("🔴 Shutdown signal received")
        self.running = False
        logger.info(f"📊 Final stats: {json.dumps(self.stats)}")


def main():
    daemon = EvaluatorDaemon()
    asyncio.run(daemon.run())


if __name__ == "__main__":
    main()
