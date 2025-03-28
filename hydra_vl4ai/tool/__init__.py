from ._base import module_registry, BaseModel
from .xvlm import XVLMModel
from .blip import BLIP2Model
from .depth import DepthEstimationModel
from .efficient_sam import EfficientSam
from .clip import CLIPModel

from ..util.console import logger

try:
    import maskrcnn_benchmark
except ImportError:
    logger.info("GLIP not installed. Skipping.")
    pass
else:
    from .glip import GLIPModel

try:
    import groundingdino
except ImportError:
    logger.info("GroundingDino not installed. Skipping.")
    pass
else:
    from .grounding_dino import GroundingDino

try:
    import llava
except ImportError:
    logger.info("LLaVA not installed. Skipping.")
    pass
else:
    from .llava import LLaVA
