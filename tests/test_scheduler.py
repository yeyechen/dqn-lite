import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.dqn import linear_schedule


def test_linear_schedule():
    eps = np.array([linear_schedule(1.0, 0.01, 300_000, t) for t in range(3_000_000)])

    assert eps[0] == 1.0
    assert eps[-1] == 0.01
    assert np.all(np.diff(eps) <= 0)
    assert np.all(eps >= 0.01)

    fig, ax = plt.subplots()
    ax.plot(eps)
    out = Path(__file__).parent / "plots" / "linear_schedule.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150)
    assert out.exists()
