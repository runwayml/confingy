"""
Main entry point for the confingy CLI.

This module combines all CLI commands into a single typer application.
"""

import typer

from confingy.cli import serialize, transpile, viz

app = typer.Typer(
    name="confingy",
    help="An implicit configuration system.",
    no_args_is_help=True,
)

# Register commands
app.command(no_args_is_help=True)(serialize.serialize)
app.command(no_args_is_help=True)(transpile.transpile)
app.command()(viz.viz)


def main() -> None:
    """Main entry point for the confingy CLI."""
    app()


if __name__ == "__main__":
    main()
