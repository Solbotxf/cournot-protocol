"""
Module 04 - Input parsing utilities for Prompt Engineer.

Provides utilities for extracting structured information from
user text input, including thresholds, dates, sources, and entities.
"""

import re
from typing import Optional


class InputParser:
    """Utilities for parsing user input into structured components."""
    
    # Common patterns for extracting information
    PRICE_PATTERN = re.compile(
        r"(?:above|below|over|under|exceed|>=?|<=?|>|<)\s*\$?([\d,]+(?:\.\d+)?)",
        re.IGNORECASE
    )
    DATE_PATTERN = re.compile(
        r"(\d{4}-\d{2}-\d{2})|"
        r"((?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2},?\s*\d{4})|"
        r"(by\s+(?:end\s+of\s+)?(?:this\s+)?(?:year|month|week|day))",
        re.IGNORECASE
    )
    TIME_PATTERN = re.compile(
        r"(\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AP]M)?)|"
        r"(midnight|noon|end\s+of\s+day)",
        re.IGNORECASE
    )
    TIMEZONE_PATTERN = re.compile(
        r"\b(UTC|EST|PST|CST|MST|EDT|PDT|CDT|MDT|GMT)\b",
        re.IGNORECASE
    )
    SOURCE_PATTERN = re.compile(
        r"(?:use|from|via|according\s+to|per)\s+([A-Za-z][A-Za-z0-9_.-]*(?:\s*(?:API|data|source)?)?)",
        re.IGNORECASE
    )
    URL_PATTERN = re.compile(
        r"https?://[^\s<>\"{}|\\^`\[\]]+",
        re.IGNORECASE
    )
    
    @classmethod
    def extract_threshold(cls, text: str) -> Optional[str]:
        """Extract numeric threshold from text."""
        match = cls.PRICE_PATTERN.search(text)
        if match:
            value = match.group(1).replace(",", "")
            return value
        return None
    
    @classmethod
    def extract_date(cls, text: str) -> Optional[str]:
        """Extract date from text."""
        match = cls.DATE_PATTERN.search(text)
        if match:
            return match.group(0)
        return None
    
    @classmethod
    def extract_timezone(cls, text: str) -> str:
        """Extract timezone from text, defaulting to UTC."""
        match = cls.TIMEZONE_PATTERN.search(text)
        if match:
            return match.group(1).upper()
        return "UTC"
    
    @classmethod
    def extract_sources(cls, text: str) -> list[str]:
        """Extract source preferences from text."""
        sources = []
        for match in cls.SOURCE_PATTERN.finditer(text):
            source = match.group(1).strip().lower()
            # Normalize common sources
            source_map = {
                "coinbase": "exchange",
                "binance": "exchange",
                "kraken": "exchange",
                "polymarket": "polymarket",
                "bls": "government",
                "sec": "government",
                "edgar": "government",
            }
            normalized = source_map.get(source, source)
            if normalized not in sources:
                sources.append(normalized)
        return sources
    
    @classmethod
    def extract_urls(cls, text: str) -> list[str]:
        """Extract explicit URLs from text."""
        return cls.URL_PATTERN.findall(text)
    
    @classmethod
    def detect_entity(cls, text: str) -> str:
        """Detect the target entity from text."""
        # Common crypto patterns
        crypto_pattern = re.compile(r"\b(BTC|ETH|SOL|XRP|ADA|DOT|DOGE|SHIB)(?:-USD)?\b", re.IGNORECASE)
        match = crypto_pattern.search(text)
        if match:
            return match.group(1).upper()
        
        # Stock patterns
        stock_pattern = re.compile(r"\$([A-Z]{1,5})\b")
        match = stock_pattern.search(text)
        if match:
            return match.group(1)
        
        # Generic entity extraction (first capitalized word sequence)
        entity_pattern = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b")
        match = entity_pattern.search(text)
        if match:
            return match.group(1)
        
        return "unknown_entity"
    
    @classmethod
    def detect_predicate(cls, text: str) -> str:
        """Detect the predicate being asserted."""
        text_lower = text.lower()
        
        predicates = {
            "close above": "price_close_above",
            "close below": "price_close_below",
            "above": "value_above",
            "below": "value_below",
            "reach": "value_reach",
            "exceed": "value_exceed",
            "announce": "event_announce",
            "release": "event_release",
            "win": "outcome_win",
            "pass": "outcome_pass",
            "fail": "outcome_fail",
            "happen": "event_occur",
            "occur": "event_occur",
        }
        
        for keyword, predicate in predicates.items():
            if keyword in text_lower:
                return predicate
        
        return "binary_outcome"