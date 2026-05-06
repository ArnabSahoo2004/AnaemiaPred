import sys
import os
import PyPDF2

pdf_path = r'e:\FRP PROJECT\Aneamia Detection\Final_FRP_Review1_Presentation.pdf'

if not os.path.exists(pdf_path):
    print(f"Error: Could not find {pdf_path}")
    sys.exit(1)

with open(pdf_path, 'rb') as f:
    reader = PyPDF2.PdfReader(f)
    for i, page in enumerate(reader.pages):
        print(f"=== Slide {i+1} ===")
        text = page.extract_text()
        if text:
            print(text.strip())
        else:
            print("[No text found on this slide]")
