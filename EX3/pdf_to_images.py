from __future__ import annotations
from pdf2image import convert_from_path
from pathlib import Path
from typing import Any
import os

class PDFConverter:
    def __init__(self) -> None:
        self.output_format: str = "png"
        self.dpi: int = 200
        
    def convert_pdf_to_images(self, pdf_path: str, output_dir: str | None = None) -> list[str]:
        pdf_file: Path = Path(pdf_path)
        
        if not pdf_file.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
        if output_dir is None:
            output_dir: str = str(pdf_file.parent / f"{pdf_file.stem}_images")
            
        output_path: Path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        try:
            pages: list[Any] = convert_from_path(pdf_path, dpi=self.dpi)
            
            image_paths: list[str] = []
            
            for i, page in enumerate(pages):
                page_num: int = i + 1
                image_filename: str = f"{pdf_file.stem}_page_{page_num:03d}.{self.output_format}"
                image_path: Path = output_path / image_filename
                
                page.save(str(image_path), self.output_format.upper())
                image_paths.append(str(image_path))
                print(f"Saved page {page_num}/{len(pages)}: {image_filename}")
                
            print(f"\nSuccessfully converted {len(pages)} pages to {output_dir}")
            return image_paths
            
        except Exception as e:
            print(f"Error converting PDF: {e}")
            raise
            
    def batch_convert(self, pdf_folder: str) -> dict[str, list[str]]:
        folder_path: Path = Path(pdf_folder)
        pdf_files: list[Path] = list(folder_path.glob("**/*.pdf"))
        
        results: dict[str, list[str]] = {}
        
        for pdf_file in pdf_files:
            print(f"\nProcessing: {pdf_file.name}")
            try:
                images: list[str] = self.convert_pdf_to_images(str(pdf_file))
                results[str(pdf_file)] = images
            except Exception as e:
                print(f"Failed to convert {pdf_file.name}: {e}")
                results[str(pdf_file)] = []
                
        return results


if __name__ == "__main__":
    converter: PDFConverter = PDFConverter()
    
    converter.convert_pdf_to_images("assignment-1.0.A.pdf", "ProblemStatement")
    converter.convert_pdf_to_images("assignment-1.0.A-guide.pdf", "ProblemGuide")