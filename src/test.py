# Simple code to see if basedpyright and ruff is working

# basedpyright should complain about the untyped parameters and return type
def test(x, y):
    return x + y


def main():
    print(test(3, 4))


if __name__ == "__main__":
    main()
