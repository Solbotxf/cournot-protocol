"""
Module 02 - Hashing Unit Tests
Tests for core/crypto/hashing.py

Tests:
- hash_bytes stability
- hash_canonical stability for dict key ordering differences
- to_hex/from_hex round trip
"""
import hashlib
import pytest

from core.crypto.hashing import (
    sha256,
    hash_bytes,
    hash_canonical,
    to_hex,
    from_hex,
    hash_concat,
)


class TestSha256:
    """Tests for sha256() function."""
    
    def test_sha256_known_value(self):
        """Test sha256 produces correct hash for known input."""
        # Known SHA-256 hash of "hello"
        expected = hashlib.sha256(b"hello").digest()
        result = sha256(b"hello")
        
        assert result == expected
        assert len(result) == 32
    
    def test_sha256_empty_bytes(self):
        """Test sha256 of empty bytes."""
        expected = hashlib.sha256(b"").digest()
        result = sha256(b"")
        
        assert result == expected
    
    def test_sha256_deterministic(self):
        """Test sha256 produces same result for same input."""
        data = b"test data for hashing"
        
        result1 = sha256(data)
        result2 = sha256(data)
        
        assert result1 == result2
    
    def test_sha256_different_inputs_different_outputs(self):
        """Test different inputs produce different hashes."""
        result1 = sha256(b"input1")
        result2 = sha256(b"input2")
        
        assert result1 != result2


class TestHashBytes:
    """Tests for hash_bytes() alias."""
    
    def test_hash_bytes_equals_sha256(self):
        """Test hash_bytes is alias for sha256."""
        data = b"test data"
        
        assert hash_bytes(data) == sha256(data)


class TestHashCanonical:
    """Tests for hash_canonical() function."""
    
    def test_hash_canonical_dict(self):
        """Test hashing a simple dictionary."""
        obj = {"key": "value", "number": 42}
        result = hash_canonical(obj)
        
        assert isinstance(result, bytes)
        assert len(result) == 32
    
    def test_hash_canonical_stable_for_key_order(self):
        """Test that dict key insertion order doesn't affect hash."""
        # Create dicts with different insertion orders
        dict1 = {"zebra": 1, "apple": 2, "mango": 3}
        dict2 = {"apple": 2, "mango": 3, "zebra": 1}
        dict3 = {"mango": 3, "zebra": 1, "apple": 2}
        
        hash1 = hash_canonical(dict1)
        hash2 = hash_canonical(dict2)
        hash3 = hash_canonical(dict3)
        
        # All should produce identical hashes due to sorted keys
        assert hash1 == hash2 == hash3
    
    def test_hash_canonical_nested_dict(self):
        """Test hashing nested dictionaries with various orders."""
        nested1 = {
            "outer_z": {"inner_b": 1, "inner_a": 2},
            "outer_a": {"inner_z": 3, "inner_m": 4},
        }
        nested2 = {
            "outer_a": {"inner_m": 4, "inner_z": 3},
            "outer_z": {"inner_a": 2, "inner_b": 1},
        }
        
        assert hash_canonical(nested1) == hash_canonical(nested2)
    
    def test_hash_canonical_list_preserves_order(self):
        """Test that list order is preserved (not sorted)."""
        list1 = {"items": [3, 1, 2]}
        list2 = {"items": [1, 2, 3]}
        
        # Different order should produce different hashes
        assert hash_canonical(list1) != hash_canonical(list2)
    
    def test_hash_canonical_deterministic_across_calls(self):
        """Test same object produces same hash across multiple calls."""
        obj = {
            "id": "test-123",
            "values": [1, 2, 3],
            "nested": {"a": "b"},
        }
        
        results = [hash_canonical(obj) for _ in range(10)]
        
        # All results should be identical
        assert all(r == results[0] for r in results)
    
    def test_hash_canonical_different_objects_different_hashes(self):
        """Test different objects produce different hashes."""
        obj1 = {"key": "value1"}
        obj2 = {"key": "value2"}
        
        assert hash_canonical(obj1) != hash_canonical(obj2)


class TestHexConversion:
    """Tests for to_hex() and from_hex() functions."""
    
    def test_to_hex_format(self):
        """Test to_hex produces correct format with 0x prefix."""
        data = bytes.fromhex("deadbeef")
        result = to_hex(data)
        
        assert result == "0xdeadbeef"
        assert result.startswith("0x")
    
    def test_to_hex_empty(self):
        """Test to_hex with empty bytes."""
        result = to_hex(b"")
        
        assert result == "0x"
    
    def test_from_hex_valid(self):
        """Test from_hex with valid input."""
        result = from_hex("0xdeadbeef")
        
        assert result == bytes.fromhex("deadbeef")
    
    def test_from_hex_empty(self):
        """Test from_hex with just prefix."""
        result = from_hex("0x")
        
        assert result == b""
    
    def test_from_hex_missing_prefix(self):
        """Test from_hex rejects input without 0x prefix."""
        with pytest.raises(ValueError, match="must start with '0x'"):
            from_hex("deadbeef")
    
    def test_from_hex_odd_length(self):
        """Test from_hex rejects odd-length hex string."""
        with pytest.raises(ValueError, match="even length"):
            from_hex("0xabc")  # 3 chars after prefix
    
    def test_from_hex_invalid_chars(self):
        """Test from_hex rejects invalid hex characters."""
        with pytest.raises(ValueError, match="Invalid hex"):
            from_hex("0xgg")
    
    def test_hex_round_trip(self):
        """Test to_hex and from_hex are inverses."""
        original = bytes.fromhex("0123456789abcdef")
        
        hex_str = to_hex(original)
        recovered = from_hex(hex_str)
        
        assert recovered == original
    
    def test_hex_round_trip_sha256(self):
        """Test round trip with actual SHA-256 hash."""
        data = b"test data"
        hash_value = sha256(data)
        
        hex_str = to_hex(hash_value)
        recovered = from_hex(hex_str)
        
        assert recovered == hash_value
        assert len(hex_str) == 2 + 64  # 0x + 64 hex chars


class TestHashConcat:
    """Tests for hash_concat() function."""
    
    def test_hash_concat_basic(self):
        """Test basic concatenation hashing."""
        left = sha256(b"left")
        right = sha256(b"right")
        
        result = hash_concat(left, right)
        expected = sha256(left + right)
        
        assert result == expected
    
    def test_hash_concat_order_matters(self):
        """Test that concatenation order affects result."""
        left = sha256(b"a")
        right = sha256(b"b")
        
        result1 = hash_concat(left, right)
        result2 = hash_concat(right, left)
        
        assert result1 != result2
    
    def test_hash_concat_deterministic(self):
        """Test hash_concat is deterministic."""
        left = sha256(b"left")
        right = sha256(b"right")
        
        results = [hash_concat(left, right) for _ in range(5)]
        
        assert all(r == results[0] for r in results)


class TestHashCanonicalEdgeCases:
    """Edge case tests for hash_canonical."""
    
    def test_hash_canonical_unicode(self):
        """Test hashing with unicode strings."""
        obj = {"message": "Hello, 世界! 🌍"}
        result = hash_canonical(obj)
        
        assert len(result) == 32
    
    def test_hash_canonical_numbers(self):
        """Test hashing with various number types."""
        obj = {
            "integer": 42,
            "negative": -17,
            "float": 3.14159,
            "zero": 0,
        }
        result = hash_canonical(obj)
        
        assert len(result) == 32
    
    def test_hash_canonical_boolean(self):
        """Test hashing with boolean values."""
        obj_true = {"flag": True}
        obj_false = {"flag": False}
        
        assert hash_canonical(obj_true) != hash_canonical(obj_false)
    
    def test_hash_canonical_empty_structures(self):
        """Test hashing empty structures."""
        empty_dict = {}
        empty_list = {"items": []}
        
        hash_dict = hash_canonical(empty_dict)
        hash_list = hash_canonical(empty_list)
        
        assert len(hash_dict) == 32
        assert len(hash_list) == 32
        assert hash_dict != hash_list


if __name__ == "__main__":
    pytest.main([__file__, "-v"])