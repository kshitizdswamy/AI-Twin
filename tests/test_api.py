import unittest
from fastapi.testclient import TestClient
from app.api import app


class TestAPIEndpoints(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")

    def test_text_analysis_endpoint(self):
        payload = {"text": "Had a fantastic day learning transformers and FastAPI!"}
        response = self.client.post("/analyze/text", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("sentiment", data)
        self.assertIn("emotion", data)

    def test_chat_endpoint(self):
        payload = {"message": "How is my screen time?"}
        response = self.client.post("/chat", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("reply", data)
        self.assertIn("bot_name", data)


if __name__ == "__main__":
    unittest.main()
