"""Supervisor agent module for claim orchestration.

This module defines a supervisor-style agent that coordinates tool-like components to
interpret claim conversations, review images, assess user history risk, and validate
evidence requirements before producing a structured claim decision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any, ClassVar, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence


@dataclass(frozen=True)
class ClaimInput:
    """Input payload for one claim row."""

    user_id: str
    image_paths: Sequence[str]
    user_claim: str
    claim_object: str


@dataclass(frozen=True)
class ClaimInterpretation:
    """Structured interpretation derived from a claim conversation."""

    claim_summary: str
    issue_type: str
    object_part: str
    severity_hint: str
    ambiguous: bool


@dataclass(frozen=True)
class ImageReview:
    """Image analysis output used by the supervisor."""

    valid_image: bool
    supporting_image_ids: tuple[str, ...]
    detected_issue_type: Optional[str]
    detected_object_part: Optional[str]
    risk_flags: tuple[str, ...]


@dataclass(frozen=True)
class HistoryRiskAssessment:
    """Risk context derived from user history."""

    risk_flags: tuple[str, ...]
    severity_modifier: Optional[str]
    rationale: str


@dataclass(frozen=True)
class EvidenceCheck:
    """Result of evidence requirement validation."""

    evidence_standard_met: bool
    evidence_standard_met_reason: str


@dataclass(frozen=True)
class SupervisorOutput:
    """Final schema required by the hackathon task."""

    evidence_standard_met: bool
    evidence_standard_met_reason: str
    risk_flags: tuple[str, ...]
    issue_type: str
    object_part: str
    claim_status: str
    claim_status_justification: str
    supporting_image_ids: tuple[str, ...]
    valid_image: bool
    severity: str


class ClaimInterpretationTool(Protocol):
    """Interface for transcript-to-claim interpretation tools."""

    def run(self, claim: ClaimInput) -> ClaimInterpretation:
        """Interpret a user claim conversation."""


class ImageReviewTool(Protocol):
    """Interface for image analysis tools."""

    def run(self, claim: ClaimInput, interpretation: ClaimInterpretation) -> ImageReview:
        """Analyze images and return evidence-relevant observations."""


class HistoryRiskTool(Protocol):
    """Interface for user history risk tools."""

    def run(self, user_id: str, user_history_row: Optional[Mapping[str, Any]]) -> HistoryRiskAssessment:
        """Assess historical risk for a user."""


class EvidenceRequirementTool(Protocol):
    """Interface for evidence requirement checkers."""

    def run(
        self,
        claim: ClaimInput,
        interpretation: ClaimInterpretation,
        image_review: ImageReview,
        requirements: Optional[Mapping[str, Any]],
    ) -> EvidenceCheck:
        """Evaluate whether the evidence standard is met."""


class SimpleClaimInterpretationTool:
    """Deterministic fallback claim interpretation tool."""

    ISSUE_KEYWORDS: ClassVar[Dict[str, tuple[str, ...]]] = {
        "dent": ("dent",),
        "scratch": ("scratch", "scrape"),
        "crack": ("crack", "broken"),
        "theft": ("stolen", "theft", "missing"),
    }
    PART_KEYWORDS: ClassVar[Dict[str, tuple[str, ...]]] = {
        "rear bumper": ("rear bumper", "back bumper"),
        "front bumper": ("front bumper",),
        "door": ("door",),
        "windshield": ("windshield", "windscreen"),
        "mirror": ("mirror",),
    }

    def run(self, claim: ClaimInput) -> ClaimInterpretation:
        transcript = claim.user_claim.strip()
        lower = transcript.lower()

        issue_type = self._match_keyword(lower, self.ISSUE_KEYWORDS, default="unknown")
        object_part = self._match_keyword(lower, self.PART_KEYWORDS, default="unknown")
        severity_hint = self._severity_from_text(lower)

        customer_messages = []
        for segment in transcript.split("|"):
            stripped = segment.strip()
            if stripped.lower().startswith("customer:") and ":" in stripped:
                customer_messages.append(stripped.split(":", 1)[1].strip())
        claim_summary = customer_messages[-1] if customer_messages else transcript
        ambiguous = issue_type == "unknown" or object_part == "unknown"

        return ClaimInterpretation(
            claim_summary=claim_summary,
            issue_type=issue_type,
            object_part=object_part,
            severity_hint=severity_hint,
            ambiguous=ambiguous,
        )

    @staticmethod
    def _match_keyword(text: str, mapping: Mapping[str, Iterable[str]], default: str) -> str:
        for label, keywords in mapping.items():
            if any(keyword in text for keyword in keywords):
                return label
        return default

    @staticmethod
    def _severity_from_text(text: str) -> str:
        if any(token in text for token in ("totaled", "severe", "major")):
            return "high"
        if any(token in text for token in ("minor", "small", "slight")):
            return "low"
        return "medium"


class SimpleImageReviewTool:
    """Deterministic fallback image review tool based on path metadata."""

    VALID_EXTENSIONS: ClassVar[set[str]] = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def run(self, claim: ClaimInput, interpretation: ClaimInterpretation) -> ImageReview:
        supporting_ids: List[str] = []
        detected_issue: Optional[str] = None
        detected_part: Optional[str] = None
        risk_flags: List[str] = []

        for image_path in claim.image_paths:
            path = Path(str(image_path).strip())
            if not path.name:
                continue
            if path.suffix.lower() in self.VALID_EXTENSIONS:
                supporting_ids.append(path.name)
                if interpretation.issue_type != "unknown" and self._filename_contains_label(path, interpretation.issue_type):
                    detected_issue = interpretation.issue_type
                if interpretation.object_part != "unknown" and self._filename_contains_label(path, interpretation.object_part):
                    detected_part = interpretation.object_part
                if "blurry" in path.name.lower() or "dark" in path.name.lower():
                    risk_flags.append("low_image_quality")

        return ImageReview(
            valid_image=bool(supporting_ids),
            supporting_image_ids=tuple(supporting_ids),
            detected_issue_type=detected_issue,
            detected_object_part=detected_part,
            risk_flags=tuple(sorted(set(risk_flags))),
        )

    @staticmethod
    def _filename_contains_label(path: Path, label: str) -> bool:
        tokens = [token for token in re.split(r"[^a-z0-9]+", path.stem.lower()) if token]
        label_tokens = [token for token in re.split(r"[^a-z0-9]+", label.lower()) if token]
        if not label_tokens:
            return False
        joined_tokens = "_".join(tokens)
        joined_label = "_".join(label_tokens)
        return joined_label in joined_tokens or all(token in tokens for token in label_tokens)


class SimpleHistoryRiskTool:
    """Deterministic fallback risk tool using user history features."""

    def run(self, user_id: str, user_history_row: Optional[Mapping[str, Any]]) -> HistoryRiskAssessment:
        if not user_history_row:
            return HistoryRiskAssessment(
                risk_flags=(),
                severity_modifier=None,
                rationale=f"No user history found for user_id={user_id}.",
            )

        risk_flags = self._parse_flags(user_history_row.get("history_flags"))
        rejected_count = self._to_int(user_history_row.get("rejected_claim"))
        manual_review_count = self._to_int(user_history_row.get("manual_review_claim"))

        if rejected_count >= 3:
            risk_flags.append("high_prior_rejections")
        if manual_review_count >= 3:
            risk_flags.append("frequent_manual_reviews")

        severity_modifier = "high" if (rejected_count >= 5 or "fraud_suspected" in risk_flags) else None
        return HistoryRiskAssessment(
            risk_flags=tuple(sorted(set(risk_flags))),
            severity_modifier=severity_modifier,
            rationale=str(user_history_row.get("history_summary") or "Risk derived from historical claim profile."),
        )

    @staticmethod
    def _parse_flags(flags: Any) -> List[str]:
        ignored_tokens = {"", "none", "null", "na", "n/a"}
        if flags is None:
            return []
        if isinstance(flags, list):
            normalized = [str(item).strip().lower() for item in flags]
            return [token for token in normalized if token not in ignored_tokens]
        if isinstance(flags, str):
            return [
                item.strip().lower()
                for item in flags.split("|")
                if item.strip().lower() not in ignored_tokens
            ]
        return []

    @staticmethod
    def _to_int(value: Any) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0


class SimpleEvidenceRequirementTool:
    """Deterministic fallback evidence requirement checker."""

    def run(
        self,
        claim: ClaimInput,
        interpretation: ClaimInterpretation,
        image_review: ImageReview,
        requirements: Optional[Mapping[str, Any]],
    ) -> EvidenceCheck:
        min_images = 1
        if requirements:
            min_images = self._to_int(requirements.get("min_images"), fallback=1)

        if not image_review.valid_image:
            return EvidenceCheck(False, "No valid images provided.")

        if len(image_review.supporting_image_ids) < min_images:
            return EvidenceCheck(
                False,
                f"Insufficient evidence images: found {len(image_review.supporting_image_ids)} but require {min_images}.",
            )

        if interpretation.ambiguous:
            return EvidenceCheck(False, "Claim is ambiguous; issue type/object part could not be confidently interpreted.")

        return EvidenceCheck(True, "Evidence requirements are satisfied by provided claim context and images.")

    @staticmethod
    def _to_int(value: Any, fallback: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return fallback


@dataclass
class SupervisorAgent:
    """Supervisor-style orchestrator for claim evaluation.

    The supervisor delegates sub-tasks to tool-like components and merges their
    outputs into the final output schema expected by the hackathon problem.
    """

    claim_tool: ClaimInterpretationTool = field(default_factory=SimpleClaimInterpretationTool)
    image_tool: ImageReviewTool = field(default_factory=SimpleImageReviewTool)
    history_tool: HistoryRiskTool = field(default_factory=SimpleHistoryRiskTool)
    evidence_tool: EvidenceRequirementTool = field(default_factory=SimpleEvidenceRequirementTool)

    def evaluate_claim(
        self,
        claim: ClaimInput,
        user_history_row: Optional[Mapping[str, Any]] = None,
        evidence_requirement_row: Optional[Mapping[str, Any]] = None,
    ) -> SupervisorOutput:
        """Evaluate a claim and return structured output for one row."""
        interpretation = self.claim_tool.run(claim)
        image_review = self.image_tool.run(claim, interpretation)
        history = self.history_tool.run(claim.user_id, user_history_row)
        evidence = self.evidence_tool.run(claim, interpretation, image_review, evidence_requirement_row)

        issue_type, object_part, conflict_flags = self._resolve_issue_and_part(interpretation, image_review)
        risk_flags = tuple(sorted(set(history.risk_flags + image_review.risk_flags + tuple(conflict_flags))))

        severity = history.severity_modifier or interpretation.severity_hint
        claim_status, justification = self._determine_status(evidence, risk_flags, history.rationale)

        return SupervisorOutput(
            evidence_standard_met=evidence.evidence_standard_met,
            evidence_standard_met_reason=evidence.evidence_standard_met_reason,
            risk_flags=risk_flags,
            issue_type=issue_type,
            object_part=object_part,
            claim_status=claim_status,
            claim_status_justification=justification,
            supporting_image_ids=image_review.supporting_image_ids,
            valid_image=image_review.valid_image,
            severity=severity,
        )

    def evaluate_claim_row(
        self,
        row: Mapping[str, Any],
        user_history_row: Optional[Mapping[str, Any]] = None,
        evidence_requirement_row: Optional[Mapping[str, Any]] = None,
    ) -> SupervisorOutput:
        """Convenience wrapper accepting dictionary-like claim rows."""
        claim = ClaimInput(
            user_id=str(row.get("user_id", "")),
            image_paths=self._normalize_image_paths(row.get("image_paths", [])),
            user_claim=str(row.get("user_claim", "")),
            claim_object=str(row.get("claim_object", "")),
        )
        return self.evaluate_claim(claim, user_history_row, evidence_requirement_row)

    @staticmethod
    def _normalize_image_paths(image_paths: Any) -> List[str]:
        if isinstance(image_paths, str):
            normalized = [segment.strip() for segment in re.split(r"[|,]", image_paths)]
            return [path for path in normalized if path]
        if isinstance(image_paths, Sequence):
            return [str(path).strip() for path in image_paths if str(path).strip()]
        return []

    @staticmethod
    def _resolve_issue_and_part(
        interpretation: ClaimInterpretation,
        image_review: ImageReview,
    ) -> tuple[str, str, List[str]]:
        flags: List[str] = []
        issue_type = interpretation.issue_type
        object_part = interpretation.object_part

        if image_review.detected_issue_type and issue_type != "unknown" and image_review.detected_issue_type != issue_type:
            flags.append("issue_image_claim_mismatch")
        if image_review.detected_object_part and object_part != "unknown" and image_review.detected_object_part != object_part:
            flags.append("object_part_image_claim_mismatch")

        if issue_type == "unknown" and image_review.detected_issue_type:
            issue_type = image_review.detected_issue_type
        if object_part == "unknown" and image_review.detected_object_part:
            object_part = image_review.detected_object_part

        return issue_type, object_part, flags

    @staticmethod
    def _determine_status(evidence: EvidenceCheck, risk_flags: Sequence[str], history_rationale: str) -> tuple[str, str]:
        if not evidence.evidence_standard_met:
            return "manual_review", f"{evidence.evidence_standard_met_reason} Sent to manual review for additional verification."

        high_risk = {"fraud_suspected", "high_prior_rejections", "issue_image_claim_mismatch", "object_part_image_claim_mismatch"}
        if any(flag in high_risk for flag in risk_flags):
            return "manual_review", f"Evidence is sufficient, but elevated risk flags require manual review. {history_rationale}"

        return "approved", "Evidence meets policy and no critical risk flags were detected."


__all__ = [
    "ClaimInput",
    "ClaimInterpretation",
    "ImageReview",
    "HistoryRiskAssessment",
    "EvidenceCheck",
    "SupervisorOutput",
    "SupervisorAgent",
    "SimpleClaimInterpretationTool",
    "SimpleImageReviewTool",
    "SimpleHistoryRiskTool",
    "SimpleEvidenceRequirementTool",
]
