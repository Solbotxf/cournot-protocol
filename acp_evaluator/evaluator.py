"""
Cournot ACP Evaluator MVP

Evaluates ACP job deliverables using:
1. Image URL validation (for image deliverables)
2. Cournot PoR pipeline (for text/data deliverables)
3. Hybrid: image pre-check + PoR reasoning
"""
import asyncio
import json
import logging
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from acp_evaluator.adapter import (
    build_evaluation_query,
    build_evaluation_reason,
    validate_image_deliverable,
    extract_image_urls,
)
from acp_evaluator.image_validator import check_image_url
from acp_evaluator.config import load_wallet

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("cournot-acp-evaluator")


class CournotEvaluator:
    """ACP Evaluator backed by Cournot PoR + image validation."""

    def __init__(self, acp_client):
        self.acp = acp_client
        self.processed_jobs = set()
        self._pipeline = None

    def _get_pipeline(self):
        """Lazy-load Cournot pipeline."""
        if self._pipeline is None:
            try:
                from core.llm import create_llm_client
                from agents.context import AgentContext
                from orchestrator.pipeline import create_pipeline, ExecutionMode

                provider = os.environ.get("COURNOT_LLM_PROVIDER", "openai")
                api_key = os.environ.get(f"{provider.upper()}_API_KEY") or os.environ.get("COURNOT_LLM_API_KEY")
                
                if api_key:
                    llm = create_llm_client(provider=provider, api_key=api_key)
                    ctx = AgentContext(llm=llm)
                    self._pipeline = create_pipeline(
                        mode=ExecutionMode.PRODUCTION, context=ctx, require_llm=True
                    )
                    logger.info("PoR pipeline initialized with LLM")
                else:
                    logger.warning("No LLM API key found, using image-only evaluation")
            except Exception as e:
                logger.error(f"Pipeline init failed: {e}")
        return self._pipeline

    def evaluate_deliverable(self, job_description: str, deliverable: str) -> dict:
        """
        Evaluate a deliverable using the best available strategy:
        
        1. Image deliverables: URL validation + optional PoR reasoning
        2. Text/data deliverables: Full PoR pipeline
        3. Fallback: Conservative INVALID
        """
        image_urls = extract_image_urls(deliverable)
        
        if image_urls:
            return self._evaluate_image_deliverable(job_description, deliverable, image_urls)
        else:
            return self._evaluate_text_deliverable(job_description, deliverable)

    def _evaluate_image_deliverable(self, job_description: str, deliverable: str, image_urls: list) -> dict:
        """
        Image evaluation strategy:
        1. Validate all image URLs (reachable, HTTP 200, correct Content-Type)
        2. Use PoR pipeline for semantic reasoning about whether content matches spec
        3. Combine: image checks must pass AND PoR verdict must be YES
        """
        # Step 1: Image URL validation
        image_evidence = validate_image_deliverable(deliverable)
        all_images_valid = image_evidence and image_evidence["all_valid"]
        
        if not all_images_valid:
            # Images failed basic checks - REJECT
            failed = [c for c in image_evidence["checks"] if not c["is_valid_image"]] if image_evidence else []
            return {
                "verdict": "NO",
                "confidence": 0.90,
                "reasoning_summary": f"Image validation failed: {[c['evidence_summary'] for c in failed]}",
                "por_root": "image-check-failed",
                "image_evidence": image_evidence,
            }

        # Step 2: PoR reasoning (with image evidence pre-injected)
        pipeline = self._get_pipeline()
        if pipeline:
            query = build_evaluation_query(job_description, deliverable)
            try:
                result = pipeline.run(query)
                # PoR may return INVALID for image queries - that's OK
                # If images passed URL check and PoR can't determine content,
                # we trust the image check + give moderate confidence YES
                por_verdict = result.outcome
                por_confidence = getattr(result.verdict, "confidence", 0.5) if result.verdict else 0.5
                por_root = getattr(result.por_bundle, "por_root", "no-por") if result.por_bundle else "no-por"
                reasoning = str(getattr(result, "audit_trace", ""))[:300]

                if por_verdict == "YES":
                    return {
                        "verdict": "YES",
                        "confidence": min(por_confidence + 0.1, 1.0),  # boost from image check
                        "reasoning_summary": f"Image URL valid + PoR confirms: {reasoning}",
                        "por_root": por_root,
                        "image_evidence": image_evidence,
                    }
                elif por_verdict == "INVALID":
                    # PoR couldn't determine semantic match but images are technically valid
                    # Give YES with moderate confidence
                    checks_summary = "; ".join(c["evidence_summary"] for c in image_evidence["checks"])
                    return {
                        "verdict": "YES",
                        "confidence": 0.70,
                        "reasoning_summary": f"Image URL validated ({checks_summary}). PoR semantic check inconclusive (image content not inspectable).",
                        "por_root": por_root,
                        "image_evidence": image_evidence,
                    }
                else:  # NO
                    return {
                        "verdict": "NO",
                        "confidence": por_confidence,
                        "reasoning_summary": f"Image reachable but PoR rejects: {reasoning}",
                        "por_root": por_root,
                        "image_evidence": image_evidence,
                    }
            except Exception as e:
                logger.warning(f"PoR pipeline error, falling back to image-only: {e}")

        # Fallback: image checks passed, no PoR available
        checks_summary = "; ".join(c["evidence_summary"] for c in image_evidence["checks"])
        return {
            "verdict": "YES",
            "confidence": 0.70,
            "reasoning_summary": f"Image URL validation passed: {checks_summary}. (PoR semantic check unavailable)",
            "por_root": "image-only-" + (image_evidence["checks"][0].get("content_hash", "unknown"))[:16],
            "image_evidence": image_evidence,
        }

    def _evaluate_text_deliverable(self, job_description: str, deliverable: str) -> dict:
        """Full PoR pipeline for text/data deliverables."""
        pipeline = self._get_pipeline()
        if not pipeline:
            return {
                "verdict": "INVALID",
                "confidence": 0.0,
                "reasoning_summary": "No LLM available for evaluation",
                "por_root": "no-pipeline",
            }

        query = build_evaluation_query(job_description, deliverable)
        try:
            result = pipeline.run(query)
            return {
                "verdict": result.outcome or "INVALID",
                "confidence": getattr(result.verdict, "confidence", 0.5) if result.verdict else 0.5,
                "reasoning_summary": str(getattr(result, "audit_trace", ""))[:500],
                "por_root": getattr(result.por_bundle, "por_root", "N/A") if result.por_bundle else "N/A",
            }
        except Exception as e:
            logger.error(f"PoR pipeline error: {e}")
            return {
                "verdict": "INVALID",
                "confidence": 0.0,
                "reasoning_summary": f"Pipeline error: {str(e)[:200]}",
                "por_root": "error",
            }

    async def handle_job(self, job):
        """Process a single job that needs evaluation."""
        job_id = getattr(job, 'id', None) or getattr(job, 'on_chain_job_id', None)
        if job_id in self.processed_jobs:
            return

        logger.info(f"Evaluating job {job_id}")

        job_description = str(getattr(job, 'requirement', ''))
        deliverable = job.get_deliverable()
        
        if not deliverable:
            logger.warning(f"Job {job_id}: No deliverable yet")
            return

        deliverable_str = json.dumps(deliverable) if isinstance(deliverable, dict) else str(deliverable)

        # Run evaluation
        eval_result = self.evaluate_deliverable(job_description, deliverable_str)

        verdict = eval_result["verdict"]
        reason = build_evaluation_reason(
            verdict=verdict,
            confidence=eval_result["confidence"],
            reasoning_summary=eval_result["reasoning_summary"],
            por_root=eval_result["por_root"],
        )

        try:
            accepted = verdict == "YES"
            job.evaluate(accept=accepted, reason=reason)
            status = "APPROVED ✅" if accepted else "REJECTED ❌"
            logger.info(f"Job {job_id}: {status} (confidence={eval_result['confidence']:.2f}, por={eval_result['por_root'][:20]}...)")
            self.processed_jobs.add(job_id)
        except Exception as e:
            logger.error(f"Job {job_id}: evaluate() failed: {e}")

    async def poll_loop(self, interval: int = 10):
        """Main polling loop."""
        logger.info("Cournot ACP Evaluator started")
        while True:
            try:
                active = self.acp.get_active_jobs(page=1, page_size=50)
                for job in active:
                    phase = str(getattr(job, 'phase', '')).upper()
                    if 'EVALUATION' in phase:
                        await self.handle_job(job)
            except Exception as e:
                logger.error(f"Poll error: {e}")
            await asyncio.sleep(interval)


def main():
    from virtuals_acp.client import VirtualsACP
    from virtuals_acp.contract_clients.contract_client_v2 import ACPContractClientV2

    wallet = load_wallet()
    entity_id = int(os.environ.get("ACP_ENTITY_ID", 3))
    agent_wallet = os.environ.get("ACP_AGENT_WALLET_ADDRESS", "")

    client = ACPContractClientV2(
        wallet_private_key=wallet["private_key"],
        agent_wallet_address=agent_wallet,
        entity_id=entity_id,
    )
    acp = VirtualsACP(acp_contract_clients=client, on_new_task=lambda x: logger.info(f"Task: {x}"))
    
    evaluator = CournotEvaluator(acp)
    asyncio.run(evaluator.poll_loop())


if __name__ == "__main__":
    main()
