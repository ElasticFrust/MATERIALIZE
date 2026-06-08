"""Convert the derivation plan markdown to PDF via weasyprint + MathJax."""
import pathlib, markdown2, weasyprint

src = pathlib.Path("/root/.claude/plans/glowing-percolating-tome.md").read_text()

html_body = markdown2.markdown(src, extras=["tables", "fenced-code-blocks"])

html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<script>
MathJax = {{
  tex: {{ inlineMath: [['$','$'],['\\\\(','\\\\)']], displayMath: [['$$','$$'],['\\\\[','\\\\]']] }},
  options: {{ skipHtmlTags: ['script','noscript','style','textarea','pre'] }}
}};
</script>
<script id="MathJax-script" async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
<style>
  body {{ font-family: 'Linux Libertine', 'Georgia', serif; font-size: 11pt;
         max-width: 900px; margin: 40px auto; line-height: 1.6; color: #111; }}
  h1 {{ font-size: 18pt; border-bottom: 2px solid #333; padding-bottom: 4px; }}
  h2 {{ font-size: 14pt; margin-top: 28px; color: #222; }}
  h3 {{ font-size: 12pt; color: #333; }}
  code, pre {{ background: #f4f4f4; font-family: monospace; font-size: 9pt; }}
  pre {{ padding: 10px; border-left: 3px solid #888; overflow-x: auto; }}
  table {{ border-collapse: collapse; width: 100%; margin: 12px 0; }}
  th, td {{ border: 1px solid #bbb; padding: 6px 10px; text-align: left; }}
  th {{ background: #eee; }}
  blockquote {{ border-left: 3px solid #aaa; margin-left: 0; padding-left: 16px;
                color: #555; font-style: italic; }}
  .boxed {{ border: 2px solid #333; padding: 8px 14px; display: inline-block;
            margin: 6px 0; }}
</style>
</head>
<body>
{html_body}
</body>
</html>"""

out = pathlib.Path("/home/user/MATERIALIZE/derivation_edge_compatibility.pdf")

# weasyprint can't render MathJax (JS), so render math as Unicode approximations
# by writing a static HTML with math already written in plain text from the markdown.
# Better: use a simple CSS-only approach without JS math rendering.

html_static = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  @page {{ size: A4; margin: 2cm 2.5cm; }}
  body {{ font-family: 'DejaVu Serif', 'Georgia', serif; font-size: 10.5pt;
         line-height: 1.65; color: #111; }}
  h1 {{ font-size: 17pt; border-bottom: 2px solid #333; padding-bottom: 6px;
        margin-top: 0; }}
  h2 {{ font-size: 13pt; margin-top: 26px; color: #222;
        border-bottom: 1px solid #ccc; padding-bottom: 2px; }}
  h3 {{ font-size: 11pt; color: #333; margin-top: 16px; }}
  code {{ background: #f4f4f4; font-family: 'DejaVu Sans Mono', monospace;
           font-size: 8.5pt; padding: 1px 4px; }}
  pre  {{ background: #f4f4f4; font-family: 'DejaVu Sans Mono', monospace;
           font-size: 8pt; padding: 10px; border-left: 3px solid #888;
           white-space: pre-wrap; word-break: break-all; }}
  table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
  th, td {{ border: 1px solid #bbb; padding: 5px 9px; }}
  th {{ background: #eee; font-weight: bold; }}
  em {{ font-style: italic; }}
  strong {{ font-weight: bold; }}
  p {{ margin: 8px 0; }}
  hr {{ border: none; border-top: 1px solid #ccc; margin: 18px 0; }}
  ul, ol {{ margin: 6px 0; padding-left: 22px; }}
  li {{ margin: 3px 0; }}
  .math {{ font-style: italic; color: #000; }}
</style>
</head>
<body>
{html_body}
</body>
</html>"""

weasyprint.HTML(string=html_static).write_pdf(str(out))
print(f"PDF written to: {out}")
