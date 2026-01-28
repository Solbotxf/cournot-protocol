"""
Module 05 - Signature Verifier

Verifies signature-based provenance proofs.
Currently a stub implementation - real cryptographic verification
can be added later.
"""

from typing import Any, Optional

from core.schemas.evidence import EvidenceItem, ProvenanceProof
from core.schemas.verification import CheckResult, VerificationResult


class SignatureVerifier:
    """
    Verifies signature-based provenance proofs.
    
    This is a stub implementation that checks for the presence
    of expected signature fields. Real cryptographic verification
    can be integrated later.
    """
    
    def __init__(
        self,
        trusted_signers: Optional[list[str]] = None,
        strict_mode: bool = True
    ):
        """
        Initialize signature verifier.
        
        Args:
            trusted_signers: List of trusted signer identifiers
            strict_mode: If True, fail on verification errors
        """
        self.trusted_signers = trusted_signers or []
        self.strict_mode = strict_mode
    
    def verify(self, item: EvidenceItem) -> VerificationResult:
        """
        Verify signature provenance for an evidence item.
        
        Args:
            item: Evidence item with signature provenance
            
        Returns:
            VerificationResult
        """
        checks: list[CheckResult] = []
        
        provenance = item.provenance
        
        # Check if this is a signature proof
        if provenance.kind != "signature":
            checks.append(CheckResult(
                check_id=f"sig_kind_{item.evidence_id}",
                ok=False,
                severity="error",
                message=f"Not a signature proof: {provenance.kind}",
                details={"kind": provenance.kind}
            ))
            return VerificationResult(ok=False, checks=checks)
        
        # Check for proof blob presence
        has_proof_blob = bool(provenance.proof_blob)
        checks.append(CheckResult(
            check_id=f"sig_blob_{item.evidence_id}",
            ok=has_proof_blob,
            severity="info" if has_proof_blob else "warn",
            message=(
                "Signature proof blob present" if has_proof_blob else
                "No signature proof blob"
            ),
            details={"has_proof_blob": has_proof_blob}
        ))
        
        # Stub: Check signature format (placeholder)
        sig_valid = self._verify_signature_stub(provenance)
        checks.append(CheckResult(
            check_id=f"sig_verify_{item.evidence_id}",
            ok=sig_valid,
            severity="info" if sig_valid else "error",
            message=(
                "Signature verification passed (stub)" if sig_valid else
                "Signature verification failed (stub)"
            ),
            details={"verified": sig_valid, "stub": True}
        ))
        
        # Check trusted signer (if configured)
        if self.trusted_signers and provenance.verifier:
            is_trusted = provenance.verifier in self.trusted_signers
            checks.append(CheckResult(
                check_id=f"sig_trusted_{item.evidence_id}",
                ok=is_trusted,
                severity="info" if is_trusted else "warn",
                message=(
                    f"Signer {provenance.verifier} is trusted" if is_trusted else
                    f"Signer {provenance.verifier} not in trusted list"
                ),
                details={
                    "verifier": provenance.verifier,
                    "is_trusted": is_trusted
                }
            ))
        
        all_ok = all(c.ok for c in checks if c.severity == "error")
        
        return VerificationResult(ok=all_ok, checks=checks)
    
    def _verify_signature_stub(self, provenance: ProvenanceProof) -> bool:
        """
        Stub signature verification.
        
        In production, this would perform actual cryptographic
        verification of the signature.
        
        Args:
            provenance: Provenance proof with signature
            
        Returns:
            True if valid (stub always returns True if proof_blob exists)
        """
        # Stub: Just check that proof_blob is present
        return bool(provenance.proof_blob)
    
    def verify_batch(
        self,
        items: list[EvidenceItem]
    ) -> VerificationResult:
        """
        Verify signatures for multiple evidence items.
        
        Args:
            items: List of evidence items
            
        Returns:
            Combined VerificationResult
        """
        all_checks: list[CheckResult] = []
        all_ok = True
        
        for item in items:
            if item.provenance.kind != "signature":
                continue
            
            result = self.verify(item)
            all_checks.extend(result.checks)
            
            if not result.ok:
                all_ok = False
        
        return VerificationResult(ok=all_ok, checks=all_checks)
    
    def extract_signature_metadata(
        self,
        provenance: ProvenanceProof
    ) -> dict[str, Any]:
        """
        Extract metadata from signature proof.
        
        Args:
            provenance: Provenance proof with signature
            
        Returns:
            Extracted metadata dict
        """
        metadata = {
            "kind": provenance.kind,
            "tier": provenance.tier,
            "verifier": provenance.verifier,
            "has_proof": bool(provenance.proof_blob),
        }
        
        # In production, parse proof_blob for more details
        if provenance.extra:
            metadata.update(provenance.extra)
        
        return metadata