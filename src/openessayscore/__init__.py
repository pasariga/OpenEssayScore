"""Independent Chinese essay scoring toolkit. No pretrained artifacts bundled."""
from .scorer import Scorer
from .service.core import ServiceError
__version__ = '0.1.0'
__all__ = ['Scorer', 'ServiceError', '__version__']
