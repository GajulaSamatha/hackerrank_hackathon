import json
from pathlib import Path
from Agents.supervisor_agent import ClaimInput, ClaimInterpretation, ImageReview

class VisionImageReviewTool:
    def __init__(self, vision_client=None):
        self.vision_client = vision_client # Replace with your actual Vision API client
        self.VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

    def run(self, claim: ClaimInput, interpretation: ClaimInterpretation) -> ImageReview:
        valid_paths = [
            p for p in claim.image_paths 
            if Path(p).suffix.lower() in self.VALID_EXTENSIONS
        ]
        
        if not valid_paths:
            return ImageReview(False, (), None, None, ("no_valid_images",))

        prompt = f"""
        You are an image assessment AI.
        The user claims there is a '{interpretation.issue_type}' on the '{interpretation.object_part}'.
        Does the image support this? Are the images high quality?
        
        Return a JSON object:
        - valid_image (boolean: true if at least one image is usable and clear)
        - supporting_image_ids (list of strings: filenames without extensions that show the damage)
        - detected_issue_type (string: the actual issue seen, or null)
        - detected_object_part (string: the part seen, or null)
        - risk_flags (list of strings: e.g. "blurry", "cropped", "glare", "wrong_object")
        """
        
        # TODO: Pass the images and prompt to your Vision model.
        # Mocking the JSON response here:
        llm_response_json = '{"valid_image": true, "supporting_image_ids": [], "detected_issue_type": "dent", "detected_object_part": "rear_bumper", "risk_flags": []}'
        data = json.loads(llm_response_json)
        
        # Automatically extract image IDs (without extensions) from paths evaluated by LLM
        supporting_ids = []
        for path in valid_paths:
            p = Path(path)
            # Add to supporting IDs if the vision model deemed it helpful
            # For hackathon safety, you can trust file lists or let the LLM return exact stems
            supporting_ids.append(p.stem) 

        return ImageReview(
            valid_image=bool(data.get("valid_image", False)),
            supporting_image_ids=tuple(supporting_ids),
            detected_issue_type=data.get("detected_issue_type"),
            detected_object_part=data.get("detected_object_part"),
            risk_flags=tuple(data.get("risk_flags", []))
        )