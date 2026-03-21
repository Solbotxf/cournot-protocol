"""Configuration for Cournot ACP Evaluator."""
import json
import os

# ACP Wallet
ACP_WALLET_PATH = os.environ.get(
    "ACP_WALLET_PATH",
    os.path.join(os.path.dirname(__file__), "../../.acp-wallet.json"),
)

def load_wallet():
    """Load wallet credentials from JSON file."""
    with open(ACP_WALLET_PATH) as f:
        return json.load(f)

# ACP SDK settings
ACP_ENTITY_ID = os.environ.get("ACP_ENTITY_ID", "")  # Set after agent registration
ACP_AGENT_WALLET_ADDRESS = os.environ.get("ACP_AGENT_WALLET_ADDRESS", "")  # Smart wallet from ACP

# Cournot pipeline settings
COURNOT_EXECUTION_MODE = os.environ.get("COURNOT_EXECUTION_MODE", "DEVELOPMENT")
