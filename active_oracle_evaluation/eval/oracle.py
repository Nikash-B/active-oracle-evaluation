"""GNU Backgammon (gnubg-nn) oracle for cubeless equity scoring.

The oracle supplies the equities that agents are scored against (PR / EPM). ``gnubg_nn`` is
imported inside ``GnubgOracle.__init__`` so that ``import active_oracle_evaluation`` stays free
of the optional dependency.

Interface used from gnubg-nn 1.1.0a8:

* ``probabilities(board, nPlies)`` -> ``(pWin, pWinGammon, pWinBG, pLoseGammon, pLoseBG)``
  for the player to act; ``pLose`` is implicit. Cubeless equity to the mover is
  ``2·pWin − 1 + pWG + pWBG − pLG − pLBG``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from active_oracle_evaluation.backgammon.core.board import Board
from active_oracle_evaluation.eval._gnubg_board import to_gnubg_board

#: Seed of gnubg-nn's RNG, set when an oracle is built (chosen for this study).
GNUBG_SEED = 0


@dataclass(frozen=True)
class OracleConfig:
    """Oracle settings, recorded with results."""

    # The suites use the 0-ply net (paper, appendix "Experimental protocol", paragraph
    # "Frozen decision suite and active selection"), and the 2-ply net is the re-score check.
    plies: int
    version: str = ""  # filled from gnubg_nn at init


def _gnubg_version() -> str:
    """Return the installed gnubg-nn version from package metadata."""
    from importlib.metadata import version

    return str(version("gnubg-nn"))


def equity_from_probs(probs) -> float:
    """Cubeless equity to the player to act from a 5-prob vector."""
    pw, pwg, pwbg, plg, plbg = (float(x) for x in probs[:5])
    return (2.0 * pw - 1.0) + pwg + pwbg - plg - plbg


class GnubgOracle:
    """Equity oracle over repo boards, backed by gnubg-nn (imported lazily)."""

    def __init__(self, cfg: OracleConfig) -> None:
        """Import gnubg-nn, seed its RNG and record the engine version in ``cfg``."""
        import gnubg_nn as g  # lazy: keep the engine import gnubg-free

        self._g = g
        g.set.seed(GNUBG_SEED)
        self.cfg = cfg if cfg.version else replace(cfg, version=_gnubg_version())

    def equity(self, board: Board) -> float:
        """Cubeless equity to the player to act from the n-ply net."""
        gb = to_gnubg_board(board)
        probs = self._g.probabilities(gb, self.cfg.plies)
        return equity_from_probs(probs)
