"""
Fallback Pattern-Based Prompt Compiler

Local deterministic fallback that extracts structure from user questions
using pattern matching. Used when LLM is unavailable.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timedelta, timezone
from typing import Any, TYPE_CHECKING
from urllib.parse import quote_plus

from core.schemas import (
    DataRequirement,
    DisputePolicy,
    MarketSpec,
    PredictionSemantics,
    PromptSpec,
    ResolutionRule,
    ResolutionRules,
    ResolutionWindow,
    SelectionPolicy,
    SourcePolicy,
    SourceTarget,
    ToolPlan,
)

if TYPE_CHECKING:
    from agents.context import AgentContext


# Common patterns for question types
PRICE_PATTERNS = [
    r"(?:will|is|does)\s+(?:the\s+)?(?:price\s+of\s+)?(\w+)\s+(?:be\s+)?(?:above|over|exceed|greater than|>)\s*\$?([\d,]+(?:\.\d+)?)",
    r"(\w+)\s+(?:above|over|exceed)\s*\$?([\d,]+(?:\.\d+)?)",
    r"(\w+)\s+(?:price|value)\s+(?:above|over|exceed)\s*\$?([\d,]+(?:\.\d+)?)",
]

EVENT_PATTERNS = [
    r"(?:will|does|is)\s+(.+?)\s+(?:win|beat|defeat)\s+(.+?)(?:\?|$)",
    r"(?:will|does|is)\s+(.+?)\s+(?:happen|occur)(?:\?|$)",
    r"(?:will)\s+(.+?)(?:\?|$)",
]

DATE_PATTERNS = [
    r"(?:by|before|on|until)\s+(\d{4}-\d{2}-\d{2})",
    r"(?:by|before|on|until)\s+([A-Za-z]+\s+\d{1,2},?\s+\d{4})",
    r"(?:by|before|on|until)\s+(?:the\s+)?end\s+of\s+(\d{4})",
    r"(?:by|before|on|until)\s+(?:the\s+)?end\s+of\s+([A-Za-z]+)\s+(\d{4})",
]

# Known data sources by category
KNOWN_SOURCES = {
    "crypto": [
        {
            "source_id": "coingecko",
            "uri_template": "https://api.coingecko.com/api/v3/simple/price?ids={asset}&vs_currencies=usd",
            "kind": "api",
        },
        {
            "source_id": "coinbase",
            "uri_template": "https://api.coinbase.com/v2/prices/{asset}-USD/spot",
            "kind": "api",
        },
    ],
    "sports": [
        {
            "source_id": "espn",
            "uri_template": "https://site.api.espn.com/apis/site/v2/sports/{sport}/{league}/scoreboard",
            "kind": "api",
        },
    ],
    "news": [
        {
            "source_id": "newsapi",
            "uri_template": "https://newsapi.org/v2/everything?q={query}",
            "kind": "api",
        },
    ],
    "polymarket": [
        {
            "source_id": "polymarket",
            "uri_template": "https://clob.polymarket.com/markets/{market_slug}",
            "kind": "api",
        },
    ],
    "generic": [
        {
            "source_id": "http",
            "uri_template": "https://www.google.com/search?q={query}",
            "kind": "web",
        },
    ],
}

# Crypto asset mappings
CRYPTO_ASSETS = {
    "btc": "bitcoin",
    "bitcoin": "bitcoin",
    "eth": "ethereum",
    "ethereum": "ethereum",
    "sol": "solana",
    "solana": "solana",
    "doge": "dogecoin",
    "dogecoin": "dogecoin",
    "xrp": "ripple",
    "ada": "cardano",
    "matic": "polygon",
    "avax": "avalanche",
    "link": "chainlink",
    "dot": "polkadot",
}


class FallbackPromptCompiler:
    """
    Pattern-based fallback compiler.
    
    Uses regex patterns and heuristics to extract structure from questions.
    Deterministic - same input always produces same output.
    """
    
    def __init__(self, strict_mode: bool = True) -> None:
        """
        Initialize compiler.
        
        Args:
            strict_mode: If True, enforce strict validation
        """
        self.strict_mode = strict_mode
    
    def compile(
        self,
        ctx: "AgentContext",
        user_input: str,
    ) -> tuple[PromptSpec, ToolPlan]:
        """
        Compile user input to PromptSpec and ToolPlan.
        
        Args:
            ctx: Agent context
            user_input: User's prediction question
        
        Returns:
            Tuple of (PromptSpec, ToolPlan)
        """
        now = ctx.now()
        
        # Normalize input
        question = user_input.strip()
        question_lower = question.lower()
        
        # Detect question type and extract info
        question_type, extracted = self._analyze_question(question_lower)
        
        # Generate deterministic market ID
        market_id = self._generate_market_id(question)
        
        # Build components based on question type
        prediction_semantics = self._build_semantics(question_type, extracted, question)
        data_requirements = self._build_requirements(question_type, extracted, question)
        resolution_rules = self._build_rules(question_type)
        allowed_sources = self._build_allowed_sources(question_type, extracted)
        
        # Build resolution window (default: next 7 days)
        deadline = extracted.get("deadline")
        if deadline:
            resolution_end = deadline
        else:
            resolution_end = now + timedelta(days=7)
        
        resolution_window = ResolutionWindow(
            start=now,
            end=resolution_end,
        )
        
        # Build MarketSpec
        market = MarketSpec(
            market_id=market_id,
            question=question,
            event_definition=self._build_event_definition(question_type, extracted, question),
            timezone="UTC",
            resolution_deadline=resolution_end,
            resolution_window=resolution_window,
            resolution_rules=resolution_rules,
            allowed_sources=allowed_sources,
            min_provenance_tier=0,
            dispute_policy=DisputePolicy(
                dispute_window_seconds=86400,
                allow_challenges=True,
            ),
        )
        
        # Build PromptSpec
        prompt_spec = PromptSpec(
            market=market,
            prediction_semantics=prediction_semantics,
            data_requirements=data_requirements,
            forbidden_behaviors=[
                "speculation_without_evidence",
                "ignore_conflicting_evidence",
            ],
            created_at=now if not self.strict_mode else None,
            extra={
                "strict_mode": self.strict_mode,
                "compiler": "fallback",
                "question_type": question_type,
                "extracted_info": extracted,
            },
        )
        
        # Build ToolPlan
        tool_plan = ToolPlan(
            plan_id=f"plan_{market_id}",
            requirements=[req.requirement_id for req in data_requirements],
            sources=list({
                target.source_id
                for req in data_requirements
                for target in req.source_targets
            }),
            min_provenance_tier=0,
            allow_fallbacks=True,
        )
        
        return prompt_spec, tool_plan
    
    def _analyze_question(self, question_lower: str) -> tuple[str, dict[str, Any]]:
        """
        Analyze question to detect type and extract information.
        
        Returns:
            Tuple of (question_type, extracted_info)
        """
        extracted: dict[str, Any] = {}
        
        # Check for price questions
        for pattern in PRICE_PATTERNS:
            match = re.search(pattern, question_lower)
            if match:
                asset = match.group(1).strip()
                threshold = match.group(2).replace(",", "")
                extracted["asset"] = asset
                extracted["threshold"] = float(threshold)
                extracted["comparison"] = "above"
                
                # Map to known crypto if applicable
                if asset.lower() in CRYPTO_ASSETS:
                    extracted["crypto_id"] = CRYPTO_ASSETS[asset.lower()]
                    return "crypto_price", extracted
                
                return "price", extracted
        
        # Check for event patterns
        for pattern in EVENT_PATTERNS:
            match = re.search(pattern, question_lower)
            if match:
                extracted["subject"] = match.group(1).strip()
                if match.lastindex and match.lastindex >= 2:
                    extracted["object"] = match.group(2).strip()
                return "event", extracted
        
        # Check for date patterns
        for pattern in DATE_PATTERNS:
            match = re.search(pattern, question_lower)
            if match:
                extracted["deadline_text"] = match.group(1)
                # Try to parse the date
                deadline = self._parse_date(match.group(1))
                if deadline:
                    extracted["deadline"] = deadline
                break
        
        # Default to generic
        return "generic", extracted
    
    def _parse_date(self, date_text: str) -> datetime | None:
        """Parse date string to datetime."""
        import calendar
        
        # Try ISO format
        try:
            return datetime.fromisoformat(date_text).replace(tzinfo=timezone.utc)
        except ValueError:
            pass
        
        # Try common formats
        formats = [
            "%B %d, %Y",
            "%B %d %Y",
            "%b %d, %Y",
            "%b %d %Y",
            "%Y",
        ]
        
        for fmt in formats:
            try:
                dt = datetime.strptime(date_text, fmt)
                if fmt == "%Y":
                    # End of year
                    dt = dt.replace(month=12, day=31)
                return dt.replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        
        return None
    
    def _build_semantics(
        self,
        question_type: str,
        extracted: dict[str, Any],
        question: str,
    ) -> PredictionSemantics:
        """Build prediction semantics from extracted info."""
        if question_type == "crypto_price":
            return PredictionSemantics(
                target_entity=extracted.get("crypto_id", extracted.get("asset", "unknown")),
                predicate=f"price {extracted.get('comparison', 'above')} threshold",
                threshold=str(extracted.get("threshold", "")),
                timeframe=extracted.get("deadline_text"),
            )
        
        if question_type == "price":
            return PredictionSemantics(
                target_entity=extracted.get("asset", "unknown"),
                predicate=f"price {extracted.get('comparison', 'above')} threshold",
                threshold=str(extracted.get("threshold", "")),
                timeframe=extracted.get("deadline_text"),
            )
        
        if question_type == "event":
            return PredictionSemantics(
                target_entity=extracted.get("subject", "event"),
                predicate=extracted.get("object", "occurs"),
                threshold=None,
                timeframe=extracted.get("deadline_text"),
            )
        
        # Generic
        return PredictionSemantics(
            target_entity="event",
            predicate="occurs",
            threshold=None,
            timeframe=extracted.get("deadline_text"),
        )
    
    def _build_requirements(
        self,
        question_type: str,
        extracted: dict[str, Any],
        question: str,
    ) -> list[DataRequirement]:
        """Build data requirements based on question type."""
        requirements: list[DataRequirement] = []
        
        if question_type == "crypto_price":
            crypto_id = extracted.get("crypto_id", "bitcoin")
            
            # Primary: CoinGecko
            requirements.append(
                DataRequirement(
                    requirement_id="req_001",
                    description=f"Get current {crypto_id} price from CoinGecko",
                    source_targets=[
                        SourceTarget(
                            source_id="coingecko",
                            uri=f"https://api.coingecko.com/api/v3/simple/price?ids={crypto_id}&vs_currencies=usd",
                            method="GET",
                            expected_content_type="json",
                        ),
                    ],
                    selection_policy=SelectionPolicy(
                        strategy="single_best",
                        min_sources=1,
                        max_sources=1,
                        quorum=1,
                    ),
                    expected_fields=["usd"],
                )
            )
            
            # Fallback: Coinbase
            asset_upper = extracted.get("asset", "btc").upper()
            requirements.append(
                DataRequirement(
                    requirement_id="req_002",
                    description=f"Get current {crypto_id} price from Coinbase (fallback)",
                    source_targets=[
                        SourceTarget(
                            source_id="coinbase",
                            uri=f"https://api.coinbase.com/v2/prices/{asset_upper}-USD/spot",
                            method="GET",
                            expected_content_type="json",
                        ),
                    ],
                    selection_policy=SelectionPolicy(
                        strategy="fallback_chain",
                        min_sources=1,
                        max_sources=1,
                        quorum=1,
                    ),
                    expected_fields=["amount"],
                )
            )
        
        elif question_type == "price":
            # Generic price lookup
            asset = extracted.get("asset", "asset")
            query = quote_plus(f"{asset} price")
            
            requirements.append(
                DataRequirement(
                    requirement_id="req_001",
                    description=f"Get current {asset} price",
                    source_targets=[
                        SourceTarget(
                            source_id="http",
                            uri=f"https://www.google.com/search?q={query}",
                            method="GET",
                            expected_content_type="html",
                        ),
                    ],
                    selection_policy=SelectionPolicy(
                        strategy="single_best",
                        min_sources=1,
                        max_sources=1,
                        quorum=1,
                    ),
                )
            )
        
        else:
            # Generic event - use news search
            query = quote_plus(question[:100])
            
            requirements.append(
                DataRequirement(
                    requirement_id="req_001",
                    description="Search for news and information about the event",
                    source_targets=[
                        SourceTarget(
                            source_id="http",
                            uri=f"https://news.google.com/search?q={query}",
                            method="GET",
                            expected_content_type="html",
                        ),
                    ],
                    selection_policy=SelectionPolicy(
                        strategy="single_best",
                        min_sources=1,
                        max_sources=3,
                        quorum=1,
                    ),
                )
            )
        
        return requirements
    
    def _build_rules(self, question_type: str) -> ResolutionRules:
        """Build resolution rules for question type."""
        base_rules = [
            ResolutionRule(
                rule_id="R_VALIDITY",
                description="Check if evidence is sufficient and valid",
                priority=100,
            ),
            ResolutionRule(
                rule_id="R_CONFLICT",
                description="Handle conflicting evidence from multiple sources",
                priority=90,
            ),
        ]
        
        if question_type in ("crypto_price", "price"):
            base_rules.append(
                ResolutionRule(
                    rule_id="R_THRESHOLD",
                    description="Compare price to threshold and return YES if above, NO if below",
                    priority=80,
                )
            )
        else:
            base_rules.append(
                ResolutionRule(
                    rule_id="R_BINARY_DECISION",
                    description="Map evidence to YES/NO based on event occurrence",
                    priority=80,
                )
            )
        
        base_rules.extend([
            ResolutionRule(
                rule_id="R_CONFIDENCE",
                description="Assign confidence score based on evidence quality",
                priority=70,
            ),
            ResolutionRule(
                rule_id="R_INVALID_FALLBACK",
                description="Return INVALID if resolution is impossible",
                priority=0,
            ),
        ])
        
        return ResolutionRules(rules=base_rules)
    
    def _build_allowed_sources(
        self,
        question_type: str,
        extracted: dict[str, Any],
    ) -> list[SourcePolicy]:
        """Build allowed sources list."""
        sources: list[SourcePolicy] = []
        
        if question_type == "crypto_price":
            sources.extend([
                SourcePolicy(source_id="coingecko", kind="api", allow=True),
                SourcePolicy(source_id="coinbase", kind="api", allow=True),
                SourcePolicy(source_id="binance", kind="api", allow=True),
            ])
        elif question_type == "price":
            sources.append(SourcePolicy(source_id="http", kind="web", allow=True))
        else:
            sources.extend([
                SourcePolicy(source_id="http", kind="web", allow=True),
                SourcePolicy(source_id="newsapi", kind="api", allow=True),
            ])
        
        return sources
    
    def _build_event_definition(
        self,
        question_type: str,
        extracted: dict[str, Any],
        question: str,
    ) -> str:
        """Build machine-evaluable event definition."""
        if question_type == "crypto_price":
            asset = extracted.get("crypto_id", extracted.get("asset", "BTC"))
            threshold = extracted.get("threshold", 0)
            comparison = extracted.get("comparison", "above")
            op = ">" if comparison == "above" else "<"
            return f"price({asset.upper()}_USD) {op} {threshold}"
        
        if question_type == "price":
            asset = extracted.get("asset", "ASSET")
            threshold = extracted.get("threshold", 0)
            comparison = extracted.get("comparison", "above")
            op = ">" if comparison == "above" else "<"
            return f"price({asset.upper()}) {op} {threshold}"
        
        # Generic - use the question itself
        return question
    
    def _generate_market_id(self, question: str) -> str:
        """Generate deterministic market ID from question."""
        hash_bytes = hashlib.sha256(question.encode()).digest()
        return f"mk_{hash_bytes[:8].hex()}"
