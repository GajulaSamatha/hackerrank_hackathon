import json
from pathlib import Path
from Agents.supervisor_agent import ClaimInput, ClaimInterpretation

class LLMClaimInterpretationTool:
    def __init__(self, llm_client=None):
        self.llm_client = llm_client # Replace with your actual LLM client

    def run(self, claim: ClaimInput) -> ClaimInterpretation:
        prompt = f"""
        You are an insurance claim interpretation expert.
        Analyze the following customer support transcript and extract the claim details.
        
        Transcript: 
        {claim.user_claim}
        
        Claimed Object: {claim.claim_object}
        
        Return a JSON object with exactly these keys:
        - claim_summary (string: a short summary of the customer's issue)
        - issue_type (string: e.g. "dent", "scratch", "crack", "water_damage", "unknown")
        - object_part (string: e.g. "rear_bumper", "front_door", "screen", "unknown")
        - severity_hint (string: "low", "medium", "high", or "unknown" based on wording)
        - ambiguous (boolean: true if the transcript is unclear or missing key details, false otherwise)
        """
        
        # TODO: Call your actual LLM here. Example: response = self.llm_client.generate(prompt)
        # Using a mock JSON response for illustration purposes:
        llm_response_json = '{"claim_summary": "Customer states the rear bumper is dented.", "issue_type": "dent", "object_part": "rear_bumper", "severity_hint": "medium", "ambiguous": false}'
        
        data = json.loads(llm_response_json)
        
        return ClaimInterpretation(
            claim_summary=data.get("claim_summary", ""),
            issue_type=data.get("issue_type", "unknown"),
            object_part=data.get("object_part", "unknown"),
            severity_hint=data.get("severity_hint", "unknown"),
            ambiguous=bool(data.get("ambiguous", True))
        )