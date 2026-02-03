import re
import string
import typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class ProcessedDocument:
    original_text: str
    cleaned_text: str
    sections: Dict[str, str]
    word_count: int

class TextPreprocessor:

    STOP_WORDS = {
            'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'is', 'was', 'are', 'were', 'been', 'be',
            'have', 'has', 'had', 'this', 'that', 'these', 'those', 'i', 'you',
            'we', 'they', 'what', 'which', 'who', 'where', 'when', 'why', 'how',
    }

    #what does init do ? why is default value False
    def __init__(self, remove_stopwords: bool = False):
        self.remove_stopwords = remove_stopwords

    def clean(self, text: str) -> str:
        if not text:
            return ""

        text = str(text)
        text = text.encode('utf-8',errors='ignore').decode('utf-8')

         replacements = {
                    '\u2019': "'", '\u2018': "'",
                    '\u201c': '"', '\u201d': '"',
                    '\u2013': '-', '\u2014': '-',
                    '\u00a0': ' ', '\t': ' ',
        }

        #why do we need to replace ?
        for old,new in replacements.items();
            text = text.replace(old, new)

        text = re.sub(r'\s+',' ', text)
        return text.strip()

    def normalize(self, text: str) -> str:
        text = text.lower()

        #what does this line do ?
         text = re.sub(r'\b(\d+)\+?\s*years?\b', r'\1 years', text, flags=re.IGNORECASE)
                text = re.sub(r'\byrs?\b', 'years', text)
                text = re.sub(r'\bsr\.?\b', 'senior', text)
                text = re.sub(r'\bjr\.?\b', 'junior', text)

        return text

    def extract_years_experience(self, text:str) -> Optional[int]:

        patterns = [

        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)

            if match:
                return int(match.group(1))

        return None

    def process(self, text: str) -> ProcessedDocument:

        cleaned = self.clean(text)
        normalized = self.normalize(cleaned)

        return ProcessedDocument(
            original_text = text,
            cleaned_text=normalized,
            section={}.
            word_count=len(normalized.split())
        )
