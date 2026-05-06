import sys
import subprocess
import os

try:
    import markdown
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "markdown", "--quiet"])
    import markdown

md_path = r"e:\FRP PROJECT\Aneamia Detection\docs\Defense_Preparation_Master.md"
html_path = r"e:\FRP PROJECT\Aneamia Detection\docs\temp.html"
pdf_path = r"e:\FRP PROJECT\Aneamia Detection\docs\Defense_Preparation_Master.pdf"

with open(md_path, "r", encoding="utf-8") as f:
    text = f.read()

# Add a little CSS for basic readability
html_content = f"""
<html>
<head>
<style>
body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 40px; }}
h1, h2, h3 {{ color: #2c3e50; border-bottom: 1px solid #ddd; padding-bottom: 5px; }}
code {{ background-color: #f4f4f4; padding: 2px 4px; border-radius: 4px; }}
blockquote {{ border-left: 4px solid #cc0000; margin: 1.5em 0px; padding: 0.5em 10px; background-color: #fdf5f5; }}
</style>
</head>
<body>
{markdown.markdown(text)}
</body>
</html>
"""

with open(html_path, "w", encoding="utf-8") as f:
    f.write(html_content)

edge_paths = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe"
]

browser_exe = None
for p in edge_paths:
    if os.path.exists(p):
        browser_exe = p
        break

if not browser_exe:
    print("Could not find Edge or Chrome to generate PDF.")
    sys.exit(1)

cmd = [
    browser_exe,
    "--headless",
    "--disable-gpu",
    f"--print-to-pdf={pdf_path}",
    f"file:///{html_path.replace(chr(92), '/')}"
]

print(f"Generating PDF using {browser_exe}...")
try:
    subprocess.run(cmd, check=True)
    if os.path.exists(html_path):
        os.remove(html_path)
    print(f"PDF successfully generated at: {pdf_path}")
except Exception as e:
    print(f"Error generating PDF: {e}")
