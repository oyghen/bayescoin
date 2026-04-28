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
