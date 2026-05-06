import fitz

doc = fitz.open("1-s2.0-S2666827025001240-main.pdf")
text = ""
for page in doc:
    text += page.get_text()

with open("paper_text.txt", "w", encoding="utf-8") as f:
    f.write(text)

print("Saved paper text to paper_text.txt")
