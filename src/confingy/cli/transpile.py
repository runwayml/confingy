"""
Transpile command for confingy CLI.

This module provides functionality to convert serialized confingy configurations
back into Python code with proper type hints and imports.
"""

import json
import sys
from pathlib import Path
from typing import Annotated, Optional

import typer

from confingy.fingy import transpile_fingy


def transpile(
    input_path: Annotated[str, typer.Argument()],
    output: Annotated[
        Optional[str], typer.Option(help="Output Python file path (default: stdout)")
    ] = None,
):
    """
    Transpile serialized confingy fingys to Python code.

    Examples:

      Transpile a JSON file to Python code
      ```
      confingy transpile fingy.json
      ```

      Write output to a file
      ```
      confingy transpile fingy.json -o fingy.py
      ```

      Transpile from stdin
      ```
      echo '{"_confingy_class": "MyClass", ...}' | confingy transpile -
      ```

      You can also pipe to ruff to format
      ```
      confingy transpile fingy.json | ruff format -
      ```
    """
    try:
        # Read input
        if input_path == "-":
            # Read from stdin
            input_data = sys.stdin.read()
            fingy_data = json.loads(input_data)
        else:
            # Read from file
            input_file = Path(input_path)
            if not input_file.exists():
                typer.echo(f"Error: Input file '{input_path}' does not exist", err=True)
                raise typer.Exit(1)

            with open(input_file) as f:
                fingy_data = json.load(f)

        # Transpile
        python_code = transpile_fingy(fingy_data)

        # Write output
        if output:
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                f.write(python_code)
            typer.echo(f"Transpiled fingy written to {output_path}", err=True)
        else:
            # Write to stdout
            typer.echo(python_code)
    except Exception as e:
        typer.echo(f"Error transpiling fingy: {e}", err=True)
        raise typer.Exit(1)
