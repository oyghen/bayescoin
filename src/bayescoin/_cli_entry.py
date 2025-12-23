import sys


def main() -> None:
    try:
        from bayescoin import cli
    except (ImportError, ModuleNotFoundError) as exc:
        print(
            "CLI dependencies missing. Use: pip install 'bayescoin[cli]'",
            file=sys.stderr,
        )
        raise SystemExit(2) from exc

    cli.main()
