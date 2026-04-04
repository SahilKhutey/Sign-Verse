from fastapi.testclient import TestClient

from api_server import server


def test_classify_sequence_endpoint(monkeypatch):
    def fake_classify(sequence):
        assert len(sequence) == 5
        return {"gesture_id": 7, "confidence": 0.91}

    monkeypatch.setattr(server.realtime, "classify_gesture_sequence", fake_classify)
    monkeypatch.setattr(server.loader, "gesture_label", lambda gid: "HELLO" if gid == 7 else None)

    client = TestClient(server.app)
    payload = {"sequence": [[0.1] * 126 for _ in range(5)]}
    resp = client.post("/gesture/classify-sequence", json=payload)

    assert resp.status_code == 200
    data = resp.json()
    assert data["gesture_id"] == 7
    assert data["gesture_label"] == "HELLO"
    assert data["sequence_length"] == 5
    assert data["confidence"] == 0.91


def test_classify_sequence_validation_error():
    client = TestClient(server.app)
    resp = client.post("/gesture/classify-sequence", json={"sequence": []})
    assert resp.status_code == 422
