from pathlib import Path
from .service.core import ScoreService
from .service.registry import register_v1


class Scorer:
    """CPU scorer using an operator-managed registry of trusted artifacts.

    Use from_v1_bundle once to register the historical, hash-pinned v1 files.
    A writing prompt is recorded as context; the v1 model does not assess relevance.
    """

    def __init__(self, model_store, *, review_bundle=None):
        self.service = ScoreService(Path(model_store), review_bundle=review_bundle)

    @classmethod
    def from_v1_bundle(cls, model_dir, *, model_store, review_bundle=None):
        register_v1(Path(model_store), Path(model_dir))
        return cls(model_store, review_bundle=review_bundle)

    def score(self, text, *, writing_prompt, authorization_scope='authorized_research',
              text_verified=True, model_version=None, include_review=False):
        return self.service.evaluate(dict(text=text, writing_prompt=writing_prompt,
            authorization_scope=authorization_scope, text_verified=text_verified,
            model_version=model_version, include_review=include_review))

    def status(self):
        return self.service.status()
