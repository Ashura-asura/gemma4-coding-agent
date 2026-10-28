"""Render paper/paper.md to paper/paper.pdf with Edge headless.

Usage:
    python paper/build_pdf.py

Requires the `markdown` package. Edge/Chromium is auto-detected; set
EDGE_PATH to override. The HTML intermediate is written next to the PDF so
the rendering can be inspected without rerunning the browser.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import markdown

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "paper.md"
HTML = HERE / "paper.html"
PDF = HERE / "paper.pdf"

CANDIDATES = (
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
)

CSS = """
@page { size: A4; margin: 17mm 15mm 18mm; }
body {
  font-family: Georgia, "Times New Roman", serif;
  font-size: 10.4pt;
  line-height: 1.42;
  color: #111;
  max-width: 100%;
}
h1 { font-size: 19pt; line-height: 1.18; margin: 0 0 6pt; }
h2 { font-size: 13.5pt; margin: 16pt 0 5pt; border-bottom: 1px solid #ddd; padding-bottom: 2pt; }
h3 { font-size: 11.3pt; margin: 12pt 0 4pt; }
p { margin: 0 0 6pt; text-align: justify; }
ul, ol { margin: 0 0 7pt 16pt; padding: 0; }
li { margin: 0 0 3pt; }
code, pre { font-family: Consolas, "Courier New", monospace; font-size: 8.6pt; }
code { background: #f4f4f4; padding: 0 2px; }
pre { background: #f6f6f6; border: 1px solid #e2e2e2; padding: 6pt; white-space: pre-wrap; }
table { border-collapse: collapse; width: 100%; margin: 7pt 0 10pt; font-size: 8.8pt; }
th, td { border: 1px solid #bbb; padding: 3.5pt 5pt; vertical-align: top; text-align: left; }
th { background: #f1f1f1; }
tr { page-break-inside: avoid; }
hr { border: none; border-top: 1px solid #ddd; margin: 12pt 0; }
a { color: #0645ad; text-decoration: none; }
strong { font-weight: 700; }
"""


def find_browser() -> str:
    override = os.environ.get("EDGE_PATH")
    if override:
        return override
    for candidate in CANDIDATES:
        if Path(candidate).exists():
            return candidate
    raise SystemExit("no Edge/Chrome found; set EDGE_PATH")


def main() -> int:
    text = SOURCE.read_text(encoding="utf-8")
    body = markdown.markdown(
        text,
        extensions=["tables", "fenced_code", "sane_lists", "toc"],
        output_format="html5",
    )
    HTML.write_text(
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Post-training Gemma 4 into an offline software-engineering agent</title>"
        f"<style>{CSS}</style></head><body>{body}</body></html>",
        encoding="utf-8",
    )
    browser = find_browser()
    cmd = [
        browser,
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        "--no-pdf-header-footer",
        f"--print-to-pdf={PDF}",
        HTML.as_uri(),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0 or not PDF.exists():
        print(result.stdout[-2000:], file=sys.stderr)
        print(result.stderr[-2000:], file=sys.stderr)
        raise SystemExit(f"PDF build failed (rc={result.returncode})")
    print(f"wrote {PDF} ({PDF.stat().st_size} bytes) from {SOURCE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
