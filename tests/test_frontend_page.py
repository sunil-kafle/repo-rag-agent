# Tests for the frontend page served by FastAPI.

from fastapi.testclient import TestClient

from app.main import app


def test_home_page_returns_html():
    client = TestClient(app)

    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "AI FAQ Assistant" in response.text
    assert "Ask me anything about the DataTalksClub/faq repository" in response.text
    assert 'id="chat-form"' in response.text
    assert 'id="chat-messages"' in response.text


def test_static_css_is_served():
    client = TestClient(app)

    response = client.get("/static/style.css")

    assert response.status_code == 200
    assert "text/css" in response.headers["content-type"]


def test_static_js_is_served():
    client = TestClient(app)

    response = client.get("/static/app.js")

    assert response.status_code == 200
    assert "javascript" in response.headers["content-type"] or "text/plain" in response.headers["content-type"]