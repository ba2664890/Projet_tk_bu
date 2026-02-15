"""Package EDA Desk Modern."""

from .app import EDADeskPro
from .ai_report import HuggingFaceReportAssistant
from .analysis import MultivariateAnalysis
from .data_sources import DataSourceManager
from .theme import Colors

__all__ = [
    "EDADeskPro",
    "HuggingFaceReportAssistant",
    "MultivariateAnalysis",
    "DataSourceManager",
    "Colors",
]
