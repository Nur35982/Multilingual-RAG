import pdfplumber
#import fitz
import re
import unicodedata
from typing import List
import nltk
from .config import settings

nltk.download('punkt')

class BengaliTextCleaner:
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize Bengali text"""
        # Normalize Unicode variants
        text = unicodedata.normalize('NFC', text)
        
        # Common Bengali cleaning
        replacements = {
                'িা': 'িয়া',
                '্ব্': '্ব',
                '্ক্ত': 'ক্ত',
                '্র্': '্র',
                '়্': '়',
                '্া': 'া',
                '্র্য': '্র্যা',
                '\uf0a7': '।' 
            }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # Remove headers/footers
        text = re.sub(r'Page \d+|©.*|\d{4}-\d{4}', '', text)
        
        # Fix line breaks and spaces
        text = re.sub(r'-\n', '', text)  # Handle hyphenated words
        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Normalize punctuation
        text = re.sub(r'[।]+', '।', text)
    
        return text.strip()

class PDFProcessor:
    def __init__(self):
        self.cleaner = BengaliTextCleaner()

    def extract_text(self, pdf_path: str) -> str:
        """Improved Bengali text extraction"""
        full_text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                # Use these settings for better Bengali text extraction
                text = page.extract_text(
                    layout=True,
                    x_tolerance=2,
                    y_tolerance=2,
                    keep_blank_chars=False
                )
                if text:
                    full_text += text + "\n"
        return self.cleaner.clean_text(full_text)

    def bengali_sentence_tokenize(self, text: str) -> List[str]:
        """Custom sentence tokenizer for Bengali"""
        sentences = re.split(r'(?<=[।?!])\s+', text)
        final_sentences = []
        for sent in sentences:
            if len(sent) > 150:
                parts = nltk.sent_tokenize(sent)
                final_sentences.extend(parts)
            else:
                final_sentences.append(sent)
        return [s.strip() for s in final_sentences if s.strip()]

    def process_pdf(self, pdf_path: str) -> List[str]:
        """Full PDF processing pipeline"""
        raw_text = self.extract_text(pdf_path)
        print(f"\nExtracted text sample:\n{raw_text[:500]}\n")  # Debug print
        sentences = self.bengali_sentence_tokenize(raw_text)
        return sentences