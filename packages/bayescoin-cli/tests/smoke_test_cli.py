import bayescoin_cli


def main():
    result = bayescoin_cli.__name__
    expected = "bayescoin_cli"
    if result == expected:
        print(f"Smoke test for {bayescoin_cli.__name__}: PASSED")
    else:
        raise RuntimeError(f"Smoke test for {bayescoin_cli.__name__}: FAILED")


if __name__ == "__main__":
    main()
