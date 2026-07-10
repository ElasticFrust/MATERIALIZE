"""
Build documentation/<name>.pdf from documentation/<name>.md.
markdown (Python) -> styled, self-contained HTML with MathJax -> headless Edge --print-to-pdf.
Usage: python documentation/_build_pdf.py MATERIALIZE
"""
import os, sys, subprocess, tempfile, time
import markdown  # available in the anaconda env

HERE = os.path.dirname(os.path.abspath(__file__))
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

name = sys.argv[1] if len(sys.argv) > 1 else "MATERIALIZE"
md_path = os.path.join(HERE, name + ".md")
pdf_path = os.path.join(HERE, name + ".pdf")

md_text = open(md_path, encoding="utf-8").read()
body = markdown.markdown(md_text, extensions=["tables", "fenced_code", "toc", "sane_lists"])

html = """<!DOCTYPE html><html><head><meta charset="utf-8">
<script>MathJax={tex:{inlineMath:[['$','$'],['\\\\(','\\\\)']],displayMath:[['$$','$$'],['\\\\[','\\\\]']]}};</script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
<style>
 @page { size: A4; margin: 1.6cm 1.8cm; }
 body { font-family: 'Segoe UI','Helvetica Neue',Arial,sans-serif; font-size: 10.5pt; line-height: 1.55;
        color:#1a1a1a; max-width: 52em; margin: 0 auto; }
 h1 { font-size: 20pt; border-bottom: 3px solid #2c3e50; padding-bottom: 6px; color:#2c3e50; }
 h2 { font-size: 14pt; margin-top: 24px; color:#2c3e50; border-bottom:1px solid #ccd; padding-bottom:3px; }
 h3 { font-size: 11.5pt; margin-top: 16px; color:#34495e; }
 code { background:#f2f4f6; font-family:'Consolas','DejaVu Sans Mono',monospace; font-size:9pt;
        padding:1px 4px; border-radius:3px; color:#c0341d; }
 pre { background:#f6f8fa; border:1px solid #e1e4e8; border-left:4px solid #2c3e50; border-radius:4px;
       padding:10px 12px; overflow-x:auto; }
 pre code { background:none; color:#24292e; font-size:8.6pt; padding:0; }
 table { border-collapse:collapse; width:100%; margin:12px 0; font-size:9.3pt; }
 th,td { border:1px solid #cbd2d9; padding:5px 9px; text-align:left; vertical-align:top; }
 th { background:#eef1f4; font-weight:600; }
 blockquote { border-left:4px solid #b0b8c0; margin:10px 0; padding:4px 14px; color:#444; background:#fafbfc; }
 a { color:#2c5aa0; text-decoration:none; }
 h1,h2,h3 { page-break-after:avoid; } pre,table { page-break-inside:avoid; }
</style></head><body>
""" + body + "</body></html>"

html_path = os.path.join(tempfile.gettempdir(), name + "_doc.html")
open(html_path, "w", encoding="utf-8").write(html)

# headless Edge: allow MathJax (CDN) to load, then print to PDF
subprocess.run([EDGE, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                "--virtual-time-budget=12000", f"--print-to-pdf={pdf_path}",
                "file:///" + html_path.replace("\\", "/")], check=True,
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
for _ in range(20):
    if os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 0:
        break
    time.sleep(0.5)
print(f"PDF: {pdf_path}  ({os.path.getsize(pdf_path)} bytes)" if os.path.exists(pdf_path) else "PDF FAILED")
