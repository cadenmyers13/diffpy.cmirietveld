import argparse

from diffpy.cmirietveld.version import __version__  # noqa


def main():
    parser = argparse.ArgumentParser(
        prog="diffpy-cmirietveld",
        description=(
            "A pack for diffpy.cmi that allows it to do Rietveld refinement "
            "of powder diffraction data. For more info, visit: "
            "https://github.com/diffpy/diffpy.cmirietveld/"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--version",
        action="store_true",
        help="Show the program's version number and exit",
    )

    args = parser.parse_args()

    if args.version:
        print(f"diffpy.cmirietveld {__version__}")
    else:
        # Default behavior when no arguments are given
        parser.print_help()


if __name__ == "__main__":
    main()
