
import click
import uvicorn
import os

from tensorus.api.security import generate_api_key
from tensorus.tensor_storage import TensorStorage

@click.group()
def cli():
    """Tensorus command-line interface."""
    pass

@cli.command()
@click.option('--host', default='127.0.0.1', help='The host to bind to.')
@click.option('--port', default=7860, help='The port to bind to.')
@click.option('--reload', is_flag=True, default=False, help='Enable auto-reload.')
def start(host, port, reload):
    """Starts the Tensorus API server."""
    click.echo(f"Starting Tensorus API server on {host}:{port}...")
    uvicorn.run("tensorus.api:app", host=host, port=port, reload=reload)

@click.group()
def dataset():
    """Manage datasets."""
    pass

@dataset.command("list")
def list_datasets():
    """Lists all available datasets."""
    storage = TensorStorage()
    datasets = storage.list_datasets()
    if datasets:
        click.echo("Available datasets:")
        for ds in datasets:
            click.echo(f"- {ds}")
    else:
        click.echo("No datasets found.")

@dataset.command("create")
@click.argument('name')
def create_dataset(name):
    """Creates a new dataset."""
    storage = TensorStorage()
    try:
        storage.create_dataset(name)
        click.echo(f"Dataset '{name}' created successfully.")
    except ValueError as e:
        click.echo(f"Error: {e}")

@dataset.command("delete")
@click.argument('name')
def delete_dataset(name):
    """Deletes a dataset."""
    storage = TensorStorage()
    try:
        storage.delete_dataset(name)
        click.echo(f"Dataset '{name}' deleted successfully.")
    except ValueError as e:
        click.echo(f"Error: {e}")

@cli.command()
def keygen():
    """Generates a new API key."""
    key = generate_api_key()
    click.echo("Generated API key:")
    click.echo(key)

cli.add_command(dataset)

if __name__ == '__main__':
    cli()
