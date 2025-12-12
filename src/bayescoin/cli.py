import matplotlib.pyplot as plt
import typer
from rich import print as rprint

import bayescoin

app = typer.Typer(add_completion=False)


@app.command()
def counts(
    successes: int,
    trials: int,
    a: float = 1.0,
    b: float = 1.0,
    hdi_level: float = 0.95,
    plot: bool = typer.Option(False, "--plot", help="Plot Beta density with HDI."),
):
    prior = bayescoin.BetaShape(a, b)
    post = prior.posterior_from_counts(successes, trials)
    rprint(post.summary(hdi_level))
    if plot:
        ax = bayescoin.plot(post.a, post.b, hdi_level)
        ax.set_title(f"Observed {successes} successes out of {trials} trials")
        ax.set_xlabel("Probability of success")
        plt.show()


def main() -> None:
    """Canonical entry point for CLI execution."""
    app()
