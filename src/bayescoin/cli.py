import matplotlib.pyplot as plt
import typer
from rich.console import Console

import bayescoin

console = Console()
app = typer.Typer(add_completion=False)


@app.callback(invoke_without_command=True)
def version(
    show: bool = typer.Option(
        False, "--version", "-v", help="Show app version and exit."
    ),
) -> None:
    if show:
        typer.echo(f"{bayescoin.__name__} {bayescoin.__version__}")
        raise typer.Exit()


@app.command()
def counts(
    successes: int,
    trials: int,
    a: float = 1.0,
    b: float = 1.0,
    hdi_level: float = 0.95,
    plot: bool = typer.Option(False, "--plot", help="Plot Beta density with HDI."),
):
    """Show updated Beta density based on observed success and trial counts."""
    prior = bayescoin.BetaShape(a, b)
    post = prior.posterior_from_counts(successes, trials)
    console.print(post.summary(hdi_level))
    if plot:
        ax = bayescoin.plot(post, hdi_level)
        success_text = "1 success" if successes == 1 else f"{successes} successes"
        trial_text = "1 trial" if trials == 1 else f"{trials} trials"
        ax.set_title(f"Observed {success_text} out of {trial_text}")
        ax.set_xlabel("Probability of success")
        plt.show()


def main() -> None:
    """Canonical entry point for CLI execution."""
    app()
