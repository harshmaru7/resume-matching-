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

    CERTIFICATIONS = {
        'aws certified', 'azure certified', 'gcp certified',
        'google cloud certified', 'tensorflow certified',
        'aws solutions architect', 'aws machine learning',
        'certified kubernetes', 'cka', 'ckad',
        'pmp', 'csm', 'professional scrum',
        'databricks certified', 'snowflake certified',
    }


    EDUCATION_PATTERNS = {
            'phd': r'\b(?:ph\.?d\.?|doctorate)\b',
            'masters': r'\b(?:m\.?s\.?|master\'?s?|mba)\b',
            'bachelors': r'\b(?:b\.?s\.?|bachelor\'?s?)\b',
    }

    def __init__(self):
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        self.skill_patterns = {}

        for skill in self.TECHNICAL_SKILLS:
            escaped = re.escape(skill)
            pattern = escaped.replace(r'\', r'[\s\-]?')
            self.skill_patterns[skill] = re.compile(rf'\b{pattern}\b', re.IGNORECASE)

        for skill in self.SOFT_SKILLS:
            escaped = re.escape(skill)
            pattern = escaped.replace(r'\',r'[\s\-]?')
            self.skill_patterns[skill] = re.compile(rf'\b{pattern}\b', re.IGNORECASE)

        self.education_regexes = {
            level: re.compile(pattern, re.IGNORECASE)
            for level, pattern in self.EDUCATION_PATTERNS.items()
        }


    def extract_technical_skilss(self, text:str) -> Set[str]:
        found = set()
        text_lower = text.lower()

        for skill in self.TECHNICAL_SKILLS:
            if skills in self.skill_patterns:
                if self.skill_patterns[skill].search(text_lower):
                    found.add(skill)
        return found

    def extract_soft_skills(self, text:str) -> Set[str]:
        found = set()
        text_lower = text.lower()

        for skill in self.SOFT_SKILLS:
            if skills in self.skill_patterns:
                if self.skill_patterns[skill].search(text_lower):
                    found.add(skill)
        return found

    def extract_certifications(self, text: str) -> Set[str]:
        found = set()
        text_lower = text.lower()

        for cert in self.CERTIFICATIONS:
            if cert in text_lower:
                found.add(cert)

        return found


    def extract_education_level(self, text:str) -> str :
        for level in ['phd','masters','bachelors']:
            if self.education_regexes[level].search(text):
                return level
        return ""

    def extract_years_experience(self, text:str) -> int:

        patterns = [
            r'(\d+)\+?\s*years?\s+(?:of\s+)?(?:experience|expertise)',
            r'(?:over|more than)\s+(\d+)\s*years?',
            r'(\d+)\+?\s*years?\s+(?:in\s+)?(?:the\s+)?(?:field|industry|role)',
            r'experience[:\s]+(\d+)\+?\s*years?',
        ]

        max_years = 0
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                years = int(match)
                max_years = max(max_years, years)
        return max_years

    def extract(self, text: str) -> ExtractedFeatures:
        return ExtractedFeatures(
            technical_skills=self.extract_technical_skilss(text),
            education_level=self.extract_education_level(text),
            certifications=self.extract_certifications(text),
            years_experience=self.extract_years_experience(text),
            keywords=self._extract_all_keywords(text)
        )

    def _extract_all_keywords(self, text: str) -> List[str]:
        all_skills = (
            list[str](self.extract_technical_skilss(text)) +
            list[str](self.extract_soft_skills(text)) +
            list[str](self.extract_certifications(text))
        )

    def calculate_skill_match_ratio(self, required_skills: Set[str], candidate_skills: Set[str]) -> float:
        matched = required_skills.intersection(candidate_skills)
        return len(matched)/ len(required_skills)

    def get_matching_skills(self, required_skills: Set[str], candidate_skills: Set[str])-> Tuple[Set[str], Set[str]]:
        matched = required_skills.intersection(candidate_skills)
        missing = required_skills - candidate_skills
        return matched,missing
