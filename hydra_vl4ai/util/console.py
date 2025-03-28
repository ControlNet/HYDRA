from rich.console import Console
import logging
from rich.logging import RichHandler

from .config import Config

console = Console()

_FORMAT = "%(message)s"

_log_level = logging.INFO if not Config.base_config.get("debug", False) else logging.DEBUG
logging.basicConfig(
    level=_log_level, format=_FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)

logger = logging.getLogger()

__all__ = ["logger", "console"]

# disable some annoying logs
logging.getLogger("openai").setLevel(logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARN)
logging.getLogger("httpcore").setLevel(logging.WARN)
logging.getLogger("urllib3").setLevel(logging.INFO)
logging.getLogger("requests").setLevel(logging.INFO)
logging.getLogger("websocket").setLevel(logging.WARN)
logging.getLogger("websockets").setLevel(logging.WARN)
logging.getLogger("matplotlib").setLevel(logging.WARN)
