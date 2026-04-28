import bayescoin


def main() -> None:
    result = bayescoin.__name__
    expected = "bayescoin"
    if result == expected:
        print(f"Smoke test for {bayescoin.__name__}: PASSED")
    else:
        raise RuntimeError(f"Smoke test for {bayescoin.__name__}: FAILED")


if __name__ == "__main__":
    main()
