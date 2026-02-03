import re
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass, field

@dataclass
class ExtractedFeatures:
    technical_skills: Set[str] = field(default_factory=set)
    soft_skils: Set[str] = field(default_factory=set)
    certifications: Set[str] = field(default_factory=set)
    education_level: str = ""
    years_experience: int = 0
    keywords: List[str] = field(default_factory=list)

class SkillExtractor:
    TECHNICAL_SKILLS = {
            'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'ruby', 'go',
            'golang', 'rust', 'scala', 'kotlin', 'swift', 'php', 'r', 'matlab',
            'julia', 'perl', 'shell', 'bash', 'sql', 'nosql',
            'machine learning', 'deep learning', 'neural networks', 'nlp',
            'natural language processing', 'computer vision', 'reinforcement learning',
            'supervised learning', 'unsupervised learning', 'feature engineering',
            'model training', 'model deployment', 'mlops', 'llm', 'large language models',
            'transformers', 'bert', 'gpt', 'generative ai', 'ai', 'artificial intelligence',
            'data science', 'data analysis', 'data engineering', 'data mining',
            'statistical analysis', 'statistics', 'predictive modeling', 'a/b testing',
            'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'sklearn', 'pandas',
            'numpy', 'scipy', 'matplotlib', 'seaborn', 'huggingface',
            'langchain', 'fastapi', 'flask', 'django', 'react', 'spacy', 'nltk',
            'aws', 'amazon web services', 'azure', 'gcp', 'google cloud',
            'docker', 'kubernetes', 'k8s', 'terraform', 'jenkins', 'ci/cd',
            'postgresql', 'mysql', 'mongodb', 'redis', 'elasticsearch',
            'snowflake', 'bigquery', 'vector databases', 'pinecone',
            'spark', 'hadoop', 'airflow', 'kafka', 'databricks',
            'git', 'github', 'gitlab',
            'rest', 'restful', 'api', 'grpc', 'json',
        }

    SOFT_SKILLS = {
        'leadership', 'communication', 'teamwork',
        'problem solving', 'analytical', 'agile',
    }

    EDUCATION_PATTERNS = {
            'phd': r'\b(?:ph\.?d\.?|doctorate)\b',
            'masters': r'\b(?:m\.?s\.?|master\'?s?|mba)\b',
            'bachelors': r'\b(?:b\.?s\.?|bachelor\'?s?)\b',
    }

    def __init__(self):
        self._compile_patterns()

    def _compile_patterns(self):
        self.education_regexes = {
            level: re.compile(pattern, re.IGNORECASE)
            for level, pattern in self.EDUCATION_PATTERNS.items()
        }

    def extract_technical_skilss(self, text:str) -> Set[str]:
        found = set()
        text_lower = text.lower()

        for skill in self.TECHNICAL_SKILLS:
            if skill in text_lower:
                found.add(skill)
        return found

    def extract_education_level(self, text:str) -> str :
        for level in ['phd','masters','bachelors']:
            if self.education_regexes[level].search(text):
                return level
        return ""

    def extract(self, text: str) -> ExtractedFeatures:
        return ExtractedFeatures(
            technical_skills=self.extract_technical_skilss(text),
            education_level=self.extract_education_level(text),
        )
