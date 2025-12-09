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
):
    prior = bayescoin.BetaShape(a, b)
    post = prior.posterior_from_counts(successes, trials)
    rprint(post.summary(hdi_level))


def main() -> None:
    """Canonical entry point for CLI execution."""
    app()
