"""Contract tests for the WSL simulation bridge."""

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_simulation_status_has_safe_unavailable_default():
    response = client.get("/api/v1/simulation/status")
    assert response.status_code == 200
    data = response.json()
    assert data["connection_state"] in {"SIMULATION_UNAVAILABLE", "READY", "RUNNING", "ERROR"}
    assert data["mission_state"] in {"IDLE", "QUEUED", "ARMING", "TAKEOFF", "TRANSIT", "INSPECTING", "RETURNING", "LANDING", "COMPLETED", "ABORTED", "ERROR"}


def test_simulation_mission_cannot_start_without_controller(monkeypatch):
    from app.services.simulation_service import simulation_service

    def unavailable(_request):
        raise RuntimeError("SIMULATION_UNAVAILABLE: launch the WSL simulation stack first")

    monkeypatch.setattr(simulation_service, "request_start", unavailable)
    response = client.post("/api/v1/simulation/mission/start", json={})
    assert response.status_code == 503
    assert response.json()["detail"].startswith("SIMULATION_UNAVAILABLE")
