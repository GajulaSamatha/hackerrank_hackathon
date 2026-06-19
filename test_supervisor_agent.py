import unittest

from agents.supervisor_agent import ClaimInput, SupervisorAgent


class TestSupervisorAgent(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = SupervisorAgent()

    def test_approves_when_evidence_is_sufficient_and_low_risk(self) -> None:
        claim = ClaimInput(
            user_id="u1",
            image_paths=["rear_bumper_dent.jpg"],
            user_claim=(
                "Customer: I found a dent in the rear bumper this morning. | "
                "Support: Thanks for sharing."
            ),
            claim_object="car",
        )

        output = self.agent.evaluate_claim(claim)

        self.assertTrue(output.evidence_standard_met)
        self.assertEqual(output.claim_status, "approved")
        self.assertEqual(output.issue_type, "dent")
        self.assertEqual(output.object_part, "rear bumper")
        self.assertTrue(output.valid_image)
        self.assertEqual(output.supporting_image_ids, ("rear_bumper_dent.jpg",))

    def test_routes_to_manual_review_when_no_valid_images(self) -> None:
        claim = ClaimInput(
            user_id="u2",
            image_paths=["notes.txt"],
            user_claim="Customer: My rear bumper has a dent.",
            claim_object="car",
        )

        output = self.agent.evaluate_claim(claim)

        self.assertFalse(output.evidence_standard_met)
        self.assertEqual(output.claim_status, "manual_review")
        self.assertFalse(output.valid_image)
        self.assertEqual(output.evidence_standard_met_reason, "No valid images provided.")

    def test_normalizes_delimited_image_paths(self) -> None:
        row = {
            "user_id": "u3",
            "image_paths": "a.jpg, b.png | c.jpeg",
            "user_claim": "Customer: rear bumper dent reported.",
            "claim_object": "car",
        }

        output = self.agent.evaluate_claim_row(row)

        self.assertTrue(output.valid_image)
        self.assertEqual(output.supporting_image_ids, ("a.jpg", "b.png", "c.jpeg"))


if __name__ == "__main__":
    unittest.main()
