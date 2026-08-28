"""vrplab -- a research platform for the volatility risk premium.

Five separate tutorial builds (an implied-vol regression dashboard, a straddle
scenario calculator, an earnings-event viewer, and a two-part Markov regime
bot) collapse into one question:

    Is event-driven implied volatility systematically higher than the volatility
    that subsequently realises, by enough to survive the spread -- and does
    conditioning on a volatility regime change the answer?

Everything in this package exists to answer that question honestly.
"""

__version__ = "0.1.0"

from .data.base import BarRequest, VolUnits, to_annualised
from .data.synthetic import SyntheticConfig, SyntheticProvider

__all__ = ["BarRequest", "VolUnits", "to_annualised",
           "SyntheticConfig", "SyntheticProvider", "__version__"]
