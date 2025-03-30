import pytest
from fastapi.testclient import TestClient
from src.api.routes import app
import os

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_analyze_endpoint():
    # Create a test audio file
    test_file_path = "test_audio.mp3"
    with open(test_file_path, "wb") as f:
        f.write(b"dummy audio content")

    try:
        with open(test_file_path, "rb") as f:
            response = client.post(
                "/analyze",
                files={"file": ("test_audio.mp3", f, "audio/mpeg")}
            )
            assert response.status_code == 200
            assert "status" in response.json()
            assert "results" in response.json()
            assert "timestamp" in response.json()
    finally:
        # Clean up test file
        if os.path.exists(test_file_path):
            os.remove(test_file_path)

def test_invalid_file():
    # Test with invalid file type
    with open("test.txt", "w") as f:
        f.write("not an audio file")

    try:
        with open("test.txt", "rb") as f:
            response = client.post(
                "/analyze",
                files={"file": ("test.txt", f, "text/plain")}
            )
            assert response.status_code == 500
    finally:
        if os.path.exists("test.txt"):
            os.remove("test.txt") 