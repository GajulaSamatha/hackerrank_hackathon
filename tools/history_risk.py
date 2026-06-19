from typing import Any, Mapping, Optional
from Agents.supervisor_agent import HistoryRiskAssessment

class AdvancedHistoryRiskTool:
    """Evaluates the user's historical behavior to flag potential risks."""
    
    def run(self, user_id: str, user_history_row: Optional[Mapping[str, Any]]) -> HistoryRiskAssessment:
        if not user_history_row:
            return HistoryRiskAssessment((), None, f"No history for user_id={user_id}.")

        risk_flags = []
        
        # Extract numerical data safely
        rejected_claims = self._to_int(user_history_row.get("rejected_claim"))
        total_claims = self._to_int(user_history_row.get("total_claims"))
        manual_reviews = self._to_int(user_history_row.get("manual_review_claim"))
        
        # Load string history flags
        raw_flags = str(user_history_row.get("history_flags", "")).lower()
        if "fraud" in raw_flags:
            risk_flags.append("possible_manipulation")
        if "frequent" in raw_flags:
            risk_flags.append("repeated_claim_behavior")

        # Business logic checks
        if rejected_claims > 2:
            risk_flags.append("high_prior_rejection_pattern")
        if manual_reviews > 2:
            risk_flags.append("frequent_manual_review_pattern")

        # Calculate modifier
        severity_modifier = "high" if "possible_manipulation" in risk_flags else None
        
        rationale = f"Analyzed {total_claims} prior claims, {rejected_claims} rejections."

        return HistoryRiskAssessment(
            risk_flags=tuple(sorted(set(risk_flags))),
            severity_modifier=severity_modifier,
            rationale=user_history_row.get("history_summary", rationale)
        )

    def _to_int(self, val: Any) -> int:
        try:
            return int(float(val))
        except (ValueError, TypeError):
            return 0