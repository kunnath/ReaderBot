from PIL import Image
import pytesseract
from fpdf import FPDF
import re

# Path to your PNG image file
image_path = './Image/image01.jpeg'

# Use pytesseract to extract text from the image
extracted_text = pytesseract.image_to_string(Image.open(image_path))

# Clean up extracted text (optional)
# Replace non-UTF-8 characters with spaces or remove them altogether
extracted_text_cleaned = re.sub(r'[^\x00-\x7F]+', ' ', extracted_text)

# Create a PDF document
pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", size=12)

# Add the cleaned extracted text to the PDF
pdf.multi_cell(200, 10, txt=extracted_text_cleaned, border=0, align='L')

# Save the PDF to a file
pdf_output = './scan/output01.pdf'
pdf.output(pdf_output)

print(f"PDF file '{pdf_output}' created successfully.")