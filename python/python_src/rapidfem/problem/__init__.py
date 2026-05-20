"""RapidFEM problem layer.

`ProblemFD` drives the frequency-domain (Nédélec-FEM) solver. `ProblemTD`
(time-domain DGTD) follows in a later phase. `Problem` is retained as an
alias of `ProblemFD` so existing code keeps working unchanged.
"""
from .fd import Adaptive, ProblemFD

# Backward-compatible alias — `rf.Problem` continues to mean the
# frequency-domain problem.
Problem = ProblemFD

__all__ = ["Problem", "ProblemFD", "Adaptive"]
