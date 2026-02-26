"""
Visualization command for confingy CLI.

This module provides functionality to start the confingy visualization server.
The visualization server requires additional dependencies from the 'viz' extra.
"""

import typer


def viz() -> None:
    """
    Start the confingy visualization server.

    This command requires the 'viz' extra dependencies.
    Install them with: uv sync --extra viz
    """
    # Protected imports for viz dependencies
    try:
        import asyncio

        import uvicorn

        from confingy.viz.server import app as viz_app
        from confingy.viz.server import populate_default_configs
    except ImportError as e:
        typer.echo(
            "Error: The visualization server requires additional dependencies.",
            err=True,
        )
        typer.echo(
            "\nPlease install them with:\n  uv sync --extra viz\n\nOr with pip:\n  pip install 'confingy[viz]'",
            err=True,
        )
        typer.echo(f"\nMissing dependency: {e.name}", err=True)
        raise typer.Exit(1)

    typer.echo("\n" + "=" * 60)
    typer.echo("🚀 Confingy Visualization Server")
    typer.echo("=" * 60)
    typer.echo("\nFeatures:")
    typer.echo("  • Select from predefined example configurations")
    typer.echo("  • Upload your own JSON configuration files")
    typer.echo("  • Compare two configurations side-by-side")
    typer.echo("  • Interactive visualization with expand/collapse")
    typer.echo("  • Export visualizations as images")
    typer.echo("\n" + "=" * 60)
    typer.echo("\nServer starting at: http://localhost:8000")
    typer.echo("\n👉 Open your browser and navigate to: http://localhost:8000")
    typer.echo("\nPress Ctrl+C to stop the server")
    typer.echo("=" * 60 + "\n")

    try:
        # Populate default configurations
        populate_default_configs()

        # Run the server
        config = uvicorn.Config(viz_app, host="0.0.0.0", port=8000, log_level="info")
        server = uvicorn.Server(config)
        asyncio.run(server.serve())

    except KeyboardInterrupt:
        typer.echo("\n\nVisualization server stopped.")
    except Exception as e:
        typer.echo(f"\nError: {e}", err=True)
        typer.echo("\nTroubleshooting:", err=True)
        typer.echo("  • Ensure port 8000 is available", err=True)
        typer.echo(
            "  • Check that all dependencies are installed: uv sync --extra viz",
            err=True,
        )
        raise typer.Exit(1)
