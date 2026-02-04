import logging
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass, field

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

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

    @property
    def model(self):
        if self._model is None:
            if not SENTENCE_TRANSFORMERS_AVAILABLE:
                raise ImportError("sentence transformer required, please install sentence-transformers")
            device = 'cuda' if self.use_gpu else 'cpu'
            self._model = SentenceTransformer(self.model_name, device = device)
        return self._model

    def _compute_semantic_similarity(self, text1:str,text:str) -> float:
        embeddings = self.model.encode([text1, text2], convert_to_numpy=True)
        similarity = np.dot(embeddings[0], embeddings[1]) / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]))
        simiarity = (similarity+1)/2
        return float(similarity)




    def score_resume(self, job_description: str, resume: strn, resume_id:str = "resume") -> MatchResult:

        return MatchResult(resume_id=resume_id,final_score=0.0,semantic_score=0.0,skill_match_score=0.0,experience_score=0.0)
