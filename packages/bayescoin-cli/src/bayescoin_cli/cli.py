import bayescoin
import matplotlib.pyplot as plt
import typer
from rich.console import Console

console = Console()
app = typer.Typer(add_completion=False)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", help="Show version and exit."),
) -> None:
    pkg_name = bayescoin.__name__
    pkg_version = typer.style(bayescoin.__version__, fg=typer.colors.CYAN)

    if version:
        typer.echo(f"{pkg_name} {pkg_version}")
        raise typer.Exit()

    if ctx.invoked_subcommand is None:
        typer.echo(f"{pkg_name} {pkg_version} ready. See --help for usage.")
        raise typer.Exit()


@app.command()
def counts(
    successes: int = typer.Argument(help="Number of successes."),
    trials: int = typer.Argument(help="Number of trials."),
    a: float = typer.Option(1.0, help="Shape parameter a (prior, > 0)."),
    b: float = typer.Option(1.0, help="Shape parameter b (prior, > 0)."),
    hdi: float = typer.Option(0.95, help="HDI credibility level (0, 1)."),
    plot: bool = typer.Option(False, "--plot", help="Plot Beta density with HDI."),
) -> None:
    """Display updated Beta density based on observed success and trial counts.

    Example:
    $ bayescoin counts 7 21
    $ bayescoin counts 7 21 --plot
    $ bayescoin counts 7 21 --plot --hdi 0.9 --a 30 --b 30
    $ bayescoin counts 7 21 --plot --hdi 0.9 --a 0.5 --b 0.5
    """
    prior = bayescoin.BetaShape(a, b)
    post = prior.posterior_from_counts(successes, trials)
    console.print(post.summary(hdi))
    if plot:
        ax = bayescoin.plot(post, hdi)
        success_text = "1 success" if successes == 1 else f"{successes} successes"
        trial_text = "1 trial" if trials == 1 else f"{trials} trials"
        ax.set_title(f"Observed {success_text} out of {trial_text}")
        ax.set_xlabel("Probability of success")
        plt.show()
