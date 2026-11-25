"""Independent Chinese essay scoring toolkit. No pretrained artifacts bundled."""
from .scorer import Scorer
from .service.core import ServiceError
__version__ = '0.1.1'
__all__ = ['Scorer', 'ServiceError', '__version__']
