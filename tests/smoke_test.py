import bayescoin


def main():
    result = bayescoin.__name__
    expected = "bayescoin"
    if result == expected:
        print("smoke test passed")
    else:
        raise RuntimeError("smoke test failed")


if __name__ == "__main__":
    main()
