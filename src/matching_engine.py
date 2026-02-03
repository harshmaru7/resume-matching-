import logging
from typing import List, Dict, Optional
from dataclasses import dataclass, field

from .preprocessor import TextPreprocessor
from .feature_extractor import SkillExtractor

@dataclass
class MatchResult:
    resume_id: str
    final_score: float
    semantic_score: float
    skill_match_score: float
    experience_score: float
    matched_skills: List[str] = field
    (default_factory=list)
    missing_skills: List[str] = field
    (default_factory=list)
    explanation: str = ""

@dataclass
class RankedResume:
    resume_id:str
    resume_text:str
    match_result: MatchResult
    rank: int = 0

class ResumeMatchingEngine:

    DEFAULT_MODEL = 'all-MiniLM-L6-v2'

    DEFAULT_WEIGHTS = {
        'semantic': 0.6,
        'skills': 0.3,
        'experience': 0.1,
    }

    def __init__(self, model_name:str = DEFAULT_MODEL, weights: Optional[Dict[str, float]]=None,):
        self.model_name = model_name
        self.weights = weights or self.DEFAULT_WEIGHTS
        self.preprocessor = TextPreprocessor()
        self.skil_extractor = SkillExtractor()
        self._model = None

    def score_resume(self, job_description: str, resume: strn, resume_id:str = "resume") -> MatchResult:

        return MatchResult(resume_id=resume_id,final_score=0.0,semantic_score=0.0,skill_match_score=0.0,experience_score=0.0)
