from typing import Any, Mapping, Optional
from Agents.supervisor_agent import ClaimInput, ClaimInterpretation, ImageReview, EvidenceCheck

class PolicyEvidenceCheckerTool:
    """Cross-references the evidence against minimum policy standards."""
    
    def run(
        self,
        claim: ClaimInput,
        interpretation: ClaimInterpretation,
        image_review: ImageReview,
        requirements: Optional[Mapping[str, Any]],
    ) -> EvidenceCheck:
        
        # Extract requirements from evidence_requirements.csv row
        min_images_required = 1
        if requirements:
            try:
                min_images_required = int(float(requirements.get("min_images", 1)))
            except (ValueError, TypeError):
                min_images_required = 1

        # Check 1: Are there enough valid images?
        num_supporting_images = len(image_review.supporting_image_ids)
        if num_supporting_images < min_images_required:
            return EvidenceCheck(
                False, 
                f"Policy requires {min_images_required} images, but only {num_supporting_images} valid supporting images were provided."
            )

        # Check 2: Are images completely unusable?
        if not image_review.valid_image or "no_valid_images" in image_review.risk_flags:
            return EvidenceCheck(False, "No usable images were submitted; unable to verify claim.")

        # Check 3: Is the claim description too ambiguous to compare against evidence?
        if interpretation.ambiguous:
            return EvidenceCheck(False, "Customer claim description is too ambiguous to firmly verify evidence.")

        # Check 4: Confirm visibility of the specified object part
        if not image_review.detected_object_part or image_review.detected_object_part == "unknown":
            return EvidenceCheck(False, f"The claimed part '{interpretation.object_part}' is not clearly visible in the evidence.")

        return EvidenceCheck(True, "Evidence meets the minimum standards and the claimed object part is visible.")