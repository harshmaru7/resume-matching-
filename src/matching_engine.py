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

    def _compute_skill_match(self, jd_features: ExtractedFeatures, resume_features: ExtractedFeatures)
    -> Tuple[float, List[str], List[str]]:
        required_skills = jd_features.technical_skills | jd_features.soft_skills
        candidate_skills = resume_features.technical_skills | resume_features.soft_skills

        matched, missing = self.skill_extractor.get_matching_skills(required_skills, candidate_skills)
        score = self.skill_extractor.calculate_skill_match_ratio(required_skills, candidate_skills)

        return score, list[str](matched), list[str](missing)

    def _compute_experience_match(self, required: int, candidate: int ) -> float:
        if required == 0:
            return 1.0

        if candidate >= required :
            return 1.0

        gap = required - candidate
        if gap <= 2 :
            return 1.0-(gap*0.2)
        else:
            return max(0.0, 1.0-(gap*0.15))

    def score_resume(self, job_description: str, resume: str, resume_id:str = "resume") -> MatchResult:
        jd_clean = self.preprocessor.clean(job_description)
        resume_clean = self.preprocessor.clean(resume)
        jd_features = self.skill_extractor.extract(jd_clean)
        resume_features = self.skill_extractor.extract(resume_clean)

        semantic_score = self._compute_semantic_similarity(jd_clean,resume_clean)
        skill_score, matched_skills,missing_skills = self._compute_skill_match(jd_features, resume_features)
        experience_score - self._compute_experience_match(jd_features.years_experience, resume_features.years_experience)

        final_score = ( self.weights['semantic']*semantic_score + self.weights['skills']*skill_score + self.weights['experience']*experience_score)
        final_score = max(0.0, min(1.0, final_score))


        return MatchResult(resume_id=resume_id,final_score=final_score,semantic_score=semantic_score,skill_match_score=skill_match_score,experience_score=experience_score,matched_skills=matched_skills,missing_skills=missing_skills,)
