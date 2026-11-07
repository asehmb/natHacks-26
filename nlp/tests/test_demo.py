import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run_demo(stdin):
    return subprocess.run([sys.executable, str(ROOT / "demo.py")], input=stdin, capture_output=True,
                          text=True, timeout=30, cwd=ROOT)


def test_demo_prints_json_actions():
    p = run_demo("grab the red screwdriver and go home\n")
    assert p.returncode == 0, p.stderr
    lines = [l.strip() for l in p.stdout.splitlines() if l.strip().startswith("{")]
    assert [json.loads(l)["action"] for l in lines] == ["pick_up", "home"]
    assert json.loads(lines[0])["target"] == {"noun": "screwdriver", "color": "red", "size": None, "position": None}
    assert "[rules]" in p.stdout


def test_demo_handles_gibberish_blank_lines_and_eof():
    p = run_demo("\n\nasdf qwer\n")
    assert p.returncode == 0, p.stderr
    assert "[none]" in p.stdout and "Traceback" not in p.stderr


def test_core_parser_has_no_third_party_imports():
    code = ("import sys; import nlp; "
            "bad = [m for m in ('numpy','cv2','onnxruntime','tokenizers','huggingface_hub','requests','httpx') "
            "if m in sys.modules]; assert not bad, bad")
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT)
    assert p.returncode == 0, p.stderr
