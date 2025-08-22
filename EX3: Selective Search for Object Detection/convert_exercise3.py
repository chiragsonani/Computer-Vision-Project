from __future__ import annotations
from pdf_to_images import PDFConverter

converter: PDFConverter = PDFConverter()
converter.convert_pdf_to_images("exercise3 (1).pdf", "problemsImages")