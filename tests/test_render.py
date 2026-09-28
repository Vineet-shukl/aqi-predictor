from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_dockerfile_listens_on_port_with_hf_fallback():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "${PORT:-7860}" in dockerfile


def test_hf_workflow_is_manual_dispatch_only():
    text = (ROOT / ".github" / "workflows" / "deploy-hf.yml").read_text(encoding="utf-8")
    header = text.split("jobs:", 1)[0]
    assert "workflow_dispatch:" in header
    assert "push:" not in header


def test_render_blueprint_is_a_free_singapore_docker_service():
    text = (ROOT / "render.yaml").read_text(encoding="utf-8")
    assert "type: web" in text
    assert "name: aqi-predictor" in text
    assert "runtime: docker" in text
    assert "plan: free" in text
    assert "region: singapore" in text
    assert "branch: main" in text
    assert "dockerfilePath: ./Dockerfile" in text
    assert "healthCheckPath: /health" in text
    assert "autoDeployTrigger: commit" in text
    assert text.count("type: web") == 1


def test_readme_documents_render_as_the_primary_host():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "https://aqi-predictor.onrender.com/docs" in readme
    assert "Vineet-shukl/aqi-predictor" in readme
    assert "render.com" in readme
    assert "New > Blueprint" in readme
    assert "Apply" in readme
    assert "may differ" in readme
    assert "about a minute" in readme
    assert "Hugging Face PRO" in readme
    assert "workflow_dispatch" in readme
    assert "billing account" in readme
    assert "pandaisop-aqi-predictor.hf.space" not in readme
    assert "auto-deployed from `main`" not in readme
