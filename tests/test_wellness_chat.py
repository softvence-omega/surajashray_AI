from fastapi.testclient import TestClient

from main import app
from app.utils.shared_state import all_info, chat_history, prompt_counters
import app.services.openai_service as oa

client = TestClient(app)


def test_clarification_flow():
    # Prepare state
    user_id = "testuser"
    room_id = "room1"
    all_info[user_id] = {"room_id": room_id}
    chat_history.pop(user_id, None)
    prompt_counters.pop(room_id, None)

    # Mock chat_with_memory for deterministic final recommendation
    original = oa.chat_with_memory
    oa.chat_with_memory = lambda q, u: "FINAL_RECOMMENDATION: Seek medical care if temp > 39C"

    try:
        # Step 1: initial symptom report -> should get first follow-up question
        r1 = client.post(
            "/api/v1/wellness/chat",
            params={"query": "I am feeling not good. My health temperature is so high", "user_id": user_id, "access_token": "token"}
        )
        assert r1.status_code == 200
        assert "temperature" in r1.json()["response"] or "how high" in r1.json()["response"].lower()

        # Step 2: answer to first question -> get second follow-up
        r2 = client.post(
            "/api/v1/wellness/chat",
            params={"query": "It's 39C and started this morning", "user_id": user_id, "access_token": "token"}
        )
        assert r2.status_code == 200
        assert "other symptoms" in r2.json()["response"].lower() or "any other" in r2.json()["response"].lower()

        # Step 3: answer second question -> should receive final recommendation (from mocked chat_with_memory)
        r3 = client.post(
            "/api/v1/wellness/chat",
            params={"query": "I have chills and headache", "user_id": user_id, "access_token": "token"}
        )
        assert r3.status_code == 200
        assert "FINAL_RECOMMENDATION" in r3.json()["response"]

    finally:
        # restore
        oa.chat_with_memory = original
        all_info.pop(user_id, None)
        chat_history.pop(user_id, None)
        prompt_counters.pop(room_id, None)
