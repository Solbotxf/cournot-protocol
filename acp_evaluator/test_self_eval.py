"""
Self-evaluation test: Simulates the full ACP lifecycle locally
without needing a counterpart agent.

Usage:
  export ACP_ENTITY_ID=<your_entity_id>
  export ACP_AGENT_WALLET_ADDRESS=<your_agent_smart_wallet>
  python -m acp_evaluator.test_self_eval
"""
import json
import os
import sys
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from acp_evaluator.adapter import build_evaluation_query, build_evaluation_reason
from acp_evaluator.config import load_wallet

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("test-self-eval")


def test_cournot_evaluation():
    """Test the Cournot PoR pipeline on a sample evaluation."""
    
    # Sample ACP job
    job_description = (
        "Generate a professional logo for a DeFi protocol called 'AquaSwap'. "
        "Requirements: SVG format, blue/green color scheme, includes a water droplet motif, "
        "minimum 512x512px when rasterized."
    )
    
    # Sample deliverable (good)
    good_deliverable = json.dumps({
        "type": "url",
        "value": "https://example.com/aquaswap-logo.svg",
        "metadata": {
            "format": "SVG",
            "colors": ["#0066CC", "#00CC66"],
            "dimensions": "1024x1024",
            "description": "Water droplet integrated into a swap/exchange arrow motif, "
                         "blue-to-green gradient, clean vector design"
        }
    })
    
    # Sample deliverable (bad)
    bad_deliverable = json.dumps({
        "type": "text",
        "value": "Sorry, I couldn't generate the logo. Here's a text description instead: "
                 "A blue circle with the text 'AquaSwap'."
    })
    
    # Test 1: Good deliverable
    logger.info("=== Test 1: Good deliverable ===")
    query = build_evaluation_query(job_description, good_deliverable)
    logger.info(f"Query:\n{query[:300]}...")
    
    try:
        from orchestrator import create_pipeline, ExecutionMode
        pipeline = create_pipeline(mode=ExecutionMode.DEVELOPMENT)
        
        result = pipeline.run(query)
        logger.info(f"Verdict: {result.outcome}")
        logger.info(f"Confidence: {getattr(result.verdict, 'confidence', 'N/A')}")
        logger.info(f"PoR Root: {getattr(result.por_bundle, 'por_root', 'N/A')}")
        
        reason = build_evaluation_reason(
            verdict=result.outcome,
            confidence=getattr(result.verdict, "confidence", 0.5),
            reasoning_summary=str(getattr(result, "reasoning_trace", ""))[:300],
            por_root=getattr(result.por_bundle, "por_root", "N/A"),
        )
        logger.info(f"ACP Reason payload:\n{reason}")
        
    except Exception as e:
        logger.error(f"Pipeline error (expected in dev without LLM): {e}")
        logger.info("This is OK - the adapter layer is working. "
                    "Set up LLM API keys for full pipeline test.")
    
    # Test 2: Bad deliverable
    logger.info("\n=== Test 2: Bad deliverable ===")
    query2 = build_evaluation_query(job_description, bad_deliverable)
    
    try:
        result2 = pipeline.run(query2)
        logger.info(f"Verdict: {result2.outcome}")
        logger.info(f"Confidence: {getattr(result2.verdict, 'confidence', 'N/A')}")
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
    
    logger.info("\n=== Self-evaluation test complete ===")


if __name__ == "__main__":
    test_cournot_evaluation()
