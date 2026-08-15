"""
Build documentation/<name>.pdf from documentation/<name>.md for each doc.
markdown (Python) -> styled self-contained HTML (local images base64-embedded) with MathJax
-> headless Edge --print-to-pdf.
Usage: python documentation/_build_pdf.py            # builds all docs in DOCS
       python documentation/_build_pdf.py NAME ...   # builds the named docs
"""
import os, sys, re, base64, mimetypes, subprocess, tempfile, time, urllib.request
import markdown  # available in the anaconda env

HERE = os.path.dirname(os.path.abspath(__file__))
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
DOCS = ["MATERIALIZE", "ARCHITECTURE", "FUTURE_DIRECTIONS"]
# MathJax must be LOADED LOCALLY and SYNCHRONOUSLY: an async CDN <script> does not finish rendering
# under Edge's --virtual-time-budget, leaving raw LaTeX in the PDF. tex-svg.js is self-contained
# (renders SVG, no separate web-font files), so caching this one file is enough for offline math.
MJ_URL = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"
MJ_CACHE = os.path.join(tempfile.gettempdir(), "mathjax_tex-svg.js")


def mathjax_local_uri():
    if not (os.path.exists(MJ_CACHE) and os.path.getsize(MJ_CACHE) > 100000):
        urllib.request.urlretrieve(MJ_URL, MJ_CACHE)
    return "file:///" + MJ_CACHE.replace("\\", "/")

CSS = """
 @page { size: A4; margin: 1.5cm 1.7cm; }
 body { font-family: 'Segoe UI','Helvetica Neue',Arial,sans-serif; font-size: 10.3pt; line-height: 1.5;
        color:#1a1a1a; max-width: 52em; margin: 0 auto; }
 h1 { font-size: 19pt; border-bottom: 3px solid #2c3e50; padding-bottom: 6px; color:#2c3e50; }
 h2 { font-size: 13.5pt; margin-top: 22px; color:#2c3e50; border-bottom:1px solid #ccd; padding-bottom:3px; }
 h3 { font-size: 11pt; margin-top: 14px; color:#34495e; }
 code { background:#f2f4f6; font-family:'Consolas','DejaVu Sans Mono',monospace; font-size:8.8pt;
        padding:1px 4px; border-radius:3px; color:#b03a20; }
 pre { background:#f6f8fa; border:1px solid #e1e4e8; border-left:4px solid #2c3e50; border-radius:4px;
       padding:9px 11px; overflow-x:auto; }
 pre code { background:none; color:#24292e; font-size:8.2pt; padding:0; }
 table { border-collapse:collapse; width:100%; margin:11px 0; font-size:9pt; }
 th,td { border:1px solid #cbd2d9; padding:4px 8px; text-align:left; vertical-align:top; }
 th { background:#eef1f4; font-weight:600; }
 blockquote { border-left:4px solid #b0b8c0; margin:9px 0; padding:3px 13px; color:#444; background:#fafbfc; }
 img { max-width:100%; height:auto; display:block; margin:10px auto; border:1px solid #e1e4e8; border-radius:4px; }
 a { color:#2c5aa0; text-decoration:none; }
 h1,h2,h3 { page-break-after:avoid; } pre,table,img { page-break-inside:avoid; }
"""


def embed_images(html):
    """Replace <img src="local/path"> with a base64 data: URI so the PDF is self-contained."""
    def repl(m):
        src = m.group(1)
        if src.startswith(("http://", "https://", "data:")):
            return m.group(0)
        path = os.path.normpath(os.path.join(HERE, src))
        if not os.path.exists(path):
            return m.group(0)
        mime = mimetypes.guess_type(path)[0] or "image/png"
        b64 = base64.b64encode(open(path, "rb").read()).decode()
        return m.group(0).replace(src, f"data:{mime};base64,{b64}")
    return re.sub(r'<img[^>]*\ssrc="([^"]+)"', repl, html)


def build(name):
    md_path, pdf_path = os.path.join(HERE, name + ".md"), os.path.join(HERE, name + ".pdf")
    md = open(md_path, encoding="utf-8").read()

    # Protect math from the markdown processor (which would otherwise eat `_`/`*`/`\` inside
    # equations and break MathJax). Hide code first so its `$` is not mistaken for math; let code go
    # through markdown normally; re-insert the RAW math into the HTML afterwards for MathJax.
    codes = []
    md = re.sub(r"```.*?```", lambda m: codes.append(m.group(0)) or f"\x01C{len(codes)-1}\x01", md, flags=re.S)
    md = re.sub(r"`[^`\n]+`", lambda m: codes.append(m.group(0)) or f"\x01C{len(codes)-1}\x01", md)
    maths = []
    md = re.sub(r"\$\$.+?\$\$", lambda m: maths.append(m.group(0)) or f"MJXEQN{len(maths)-1}ENDMJX", md, flags=re.S)
    md = re.sub(r"\$[^$\n]+?\$", lambda m: maths.append(m.group(0)) or f"MJXEQN{len(maths)-1}ENDMJX", md)
    for i, c in enumerate(codes):                                  # restore code -> markdown renders it
        md = md.replace(f"\x01C{i}\x01", c)

    body = markdown.markdown(md, extensions=["tables", "fenced_code", "toc", "sane_lists"])
    for j, x in enumerate(maths):                                 # restore RAW math for MathJax
        body = body.replace(f"MJXEQN{j}ENDMJX", x)
    body = embed_images(body)
    html = ('<!DOCTYPE html><html><head><meta charset="utf-8">'
            '<script>MathJax={tex:{inlineMath:[["$","$"],["\\\\(","\\\\)"]],'
            'displayMath:[["$$","$$"],["\\\\[","\\\\]"]]},svg:{fontCache:"none"}};</script>'
            f'<script src="{mathjax_local_uri()}"></script>'
            f'<style>{CSS}</style></head><body>{body}</body></html>')
    html_path = os.path.join(tempfile.gettempdir(), name + "_doc.html")
    open(html_path, "w", encoding="utf-8").write(html)
    tmp_pdf = os.path.join(tempfile.gettempdir(), name + "_out.pdf")   # print here, then move into place
    if os.path.exists(tmp_pdf):
        os.remove(tmp_pdf)
    subprocess.run([EDGE, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    "--run-all-compositor-stages-before-draw", "--virtual-time-budget=90000",
                    f"--print-to-pdf={tmp_pdf}", "file:///" + html_path.replace("\\", "/")],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(40):
        if os.path.exists(tmp_pdf) and os.path.getsize(tmp_pdf) > 0:
            break
        time.sleep(0.5)
    if not (os.path.exists(tmp_pdf) and os.path.getsize(tmp_pdf) > 0):
        print(f"FAILED: {name}.pdf (Edge produced nothing)"); return
    try:
        os.replace(tmp_pdf, pdf_path)                                  # atomic when possible
        print(f"PDF: {name}.pdf  ({os.path.getsize(pdf_path)//1024} KB)")
    except PermissionError:
        alt = os.path.join(HERE, name + "_new.pdf"); os.replace(tmp_pdf, alt)
        print(f"WARNING: {name}.pdf is locked (close your PDF viewer). Wrote {name}_new.pdf instead.")


if __name__ == "__main__":
    for n in (sys.argv[1:] or DOCS):
        build(n)
