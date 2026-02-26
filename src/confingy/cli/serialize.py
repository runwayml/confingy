"""
Serialize command for confingy CLI.

This module provides functionality to serialize Python configuration objects to JSON.
"""

import inspect
import json
from typing import Annotated

import typer

from confingy import serialize_fingy
from confingy.utils.imports import load_variable_from_file


def serialize(
    file_spec: Annotated[
        str,
        typer.Argument(
            help="File path or dotted module path with optional variable name "
            "(e.g. path/to/file.py, path.to.module, path/to/file.py::var_name). "
            "If the resolved variable is a function, it is called automatically. "
            "Defaults to loading 'config'."
        ),
    ],
    output: Annotated[
        str | None,
        typer.Option(
            help="Output JSON file path (optional, prints to stdout if not provided)"
        ),
    ] = None,
) -> None:
    """
    Serialize a configuration object to JSON.
    """
    try:
        config = load_variable_from_file(file_spec)
        if inspect.isfunction(config):
            config = config()
        serialized = serialize_fingy(config)
        json_output = json.dumps(serialized)

        if output:
            with open(output, "w") as f:
                f.write(json_output)
            typer.echo(f"Configuration serialized to {output}")
        else:
            # Print to stdout
            typer.echo(json_output)
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise e
