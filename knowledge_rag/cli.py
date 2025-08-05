#!/usr/bin/env python3
"""
Command Line Interface for Knowledge RAG File Management System
"""

import click
import logging
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich import print as rprint

from src.file_management import FileManager
from config import config


# Initialize console for rich output
console = Console()


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, config.logging.level.upper()),
        format=config.logging.log_format,
        filename=config.logging.log_file
    )


@click.group()
@click.option('--root-folder', '-r', 
              default=config.file_manager.root_folder,
              help='Root folder to scan for documents')
@click.option('--db-path', '-d',
              default=config.file_manager.db_path,
              help='Path to SQLite database')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.pass_context
def cli(ctx, root_folder, db_path, verbose):
    """Knowledge RAG File Management System CLI"""
    ctx.ensure_object(dict)
    
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize FileManager
    ctx.obj['file_manager'] = FileManager(
        root_folder=root_folder,
        db_path=db_path
    )
    
    setup_logging()


@cli.command()
@click.option('--recursive/--no-recursive', default=True,
              help='Scan subdirectories recursively')
@click.pass_context
def scan(ctx, recursive):
    """Scan for supported files in the root folder"""
    file_manager = ctx.obj['file_manager']
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Scanning files...", total=None)
        files = file_manager.scan_files(recursive=recursive)
        progress.update(task, completed=True)
    
    if files:
        table = Table(title="Found Files")
        table.add_column("File Name", style="cyan")
        table.add_column("Path", style="blue")
        table.add_column("Size", style="green")
        table.add_column("Format", style="yellow")
        
        for file_path in files:
            size = file_path.stat().st_size
            size_str = f"{size:,} bytes"
            if size > 1024:
                size_str = f"{size/1024:.1f} KB"
            if size > 1024*1024:
                size_str = f"{size/(1024*1024):.1f} MB"
                
            format_ext = file_path.suffix.upper().lstrip('.')
            
            table.add_row(
                file_path.name,
                str(file_path.relative_to(file_manager.root_folder)),
                size_str,
                format_ext
            )
        
        console.print(table)
        rprint(f"\n[green]Found {len(files)} supported files[/green]")
    else:
        rprint("[yellow]No supported files found[/yellow]")


@cli.command()
@click.option('--recursive/--no-recursive', default=True,
              help='Scan subdirectories recursively')
@click.pass_context
def process(ctx, recursive):
    """Process all unprocessed files"""
    file_manager = ctx.obj['file_manager']
    
    # Get unprocessed files
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Finding unprocessed files...", total=None)
        unprocessed_files = file_manager.get_unprocessed_files(recursive=recursive)
        progress.update(task, completed=True)
    
    if not unprocessed_files:
        rprint("[green]All files are already processed![/green]")
        return
    
    rprint(f"[yellow]Found {len(unprocessed_files)} unprocessed files[/yellow]")
    
    # Process files with progress bar
    with Progress(console=console) as progress:
        task = progress.add_task("Processing files...", total=len(unprocessed_files))
        
        stats = {'success': 0, 'failed': 0, 'skipped': 0}
        
        for file_path in unprocessed_files:
            progress.update(task, description=f"Processing {file_path.name}")
            
            result = file_manager.process_file(file_path)
            
            if result:
                if result.status.value == 'completed':
                    stats['success'] += 1
                    rprint(f"[green]✓[/green] {file_path.name} - {result.episodes_created} episodes")
                elif result.status.value == 'failed':
                    stats['failed'] += 1
                    rprint(f"[red]✗[/red] {file_path.name} - {result.error_message}")
                else:
                    stats['skipped'] += 1
                    rprint(f"[yellow]⚠[/yellow] {file_path.name} - skipped")
            else:
                stats['skipped'] += 1
                rprint(f"[yellow]⚠[/yellow] {file_path.name} - skipped")
            
            progress.advance(task)
    
    # Display final stats
    panel = Panel(
        f"[green]Successfully processed: {stats['success']}[/green]\n"
        f"[red]Failed: {stats['failed']}[/red]\n"
        f"[yellow]Skipped: {stats['skipped']}[/yellow]",
        title="Processing Complete",
        title_align="center"
    )
    console.print(panel)


@cli.command()
@click.pass_context
def status(ctx):
    """Show processing status and statistics"""
    file_manager = ctx.obj['file_manager']
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Getting status...", total=None)
        status_info = file_manager.get_processing_status()
        progress.update(task, completed=True)
    
    # Display general info
    rprint(f"[blue]Root Folder:[/blue] {status_info['root_folder']}")
    rprint(f"[blue]Total Files in Folder:[/blue] {status_info['total_files_in_folder']}")
    rprint(f"[blue]Supported Extensions:[/blue] {', '.join(status_info['supported_extensions'])}")
    rprint()
    
    # Display tracker stats
    tracker_stats = status_info.get('tracker_stats', {})
    if tracker_stats:
        table = Table(title="Processing Statistics")
        table.add_column("Status", style="cyan")
        table.add_column("Count", style="green")
        
        for status, count in tracker_stats.items():
            if status not in ['total_episodes', 'total_entities', 'total_relationships']:
                table.add_row(status.title(), str(count))
        
        console.print(table)
        
        if 'total_episodes' in tracker_stats:
            rprint(f"\n[green]Total Episodes Created: {tracker_stats['total_episodes']}[/green]")
    
    # Display parser stats
    parser_stats = status_info.get('parser_stats', {})
    if parser_stats and 'formats_per_parser' in parser_stats:
        rprint("\n[blue]Available Parsers:[/blue]")
        for parser, format_count in parser_stats['formats_per_parser'].items():
            rprint(f"  • {parser}: {format_count} formats")


@cli.command()
@click.pass_context
def reprocess_failed(ctx):
    """Reprocess files that previously failed"""
    file_manager = ctx.obj['file_manager']
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Reprocessing failed files...", total=None)
        stats = file_manager.reprocess_failed_files()
        progress.update(task, completed=True)
    
    if stats['total_failed'] == 0:
        rprint("[green]No failed files found![/green]")
    else:
        panel = Panel(
            f"[blue]Total Failed Files: {stats['total_failed']}[/blue]\n"
            f"[green]Successfully Reprocessed: {stats['reprocessed_successfully']}[/green]\n"
            f"[red]Still Failing: {stats['still_failing']}[/red]",
            title="Reprocessing Results",
            title_align="center"
        )
        console.print(panel)


@cli.command()
@click.option('--max-age-days', default=7, help='Maximum age in days for failed records to keep')
@click.pass_context
def cleanup(ctx, max_age_days):
    """Clean up old failed file records"""
    file_manager = ctx.obj['file_manager']
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Cleaning up old records...", total=None)
        deleted_count = file_manager.cleanup_tracker(max_age_days=max_age_days)
        progress.update(task, completed=True)
    
    rprint(f"[green]Cleaned up {deleted_count} old failed records[/green]")


@cli.command()
@click.argument('file_path', type=click.Path(exists=True))
@click.pass_context
def process_single(ctx, file_path):
    """Process a single file"""
    file_manager = ctx.obj['file_manager']
    file_path = Path(file_path)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(f"Processing {file_path.name}...", total=None)
        result = file_manager.process_file(file_path)
        progress.update(task, completed=True)
    
    if result:
        if result.status.value == 'completed':
            panel = Panel(
                f"[green]Status: {result.status.value.title()}[/green]\n"
                f"[blue]Episodes Created: {result.episodes_created}[/blue]\n"
                f"[blue]File Size: {result.metadata.file_size:,} bytes[/blue]\n"
                f"[blue]Format: {result.metadata.file_format.value.upper()}[/blue]",
                title=f"Processing Result: {file_path.name}",
                title_align="center"
            )
        else:
            panel = Panel(
                f"[red]Status: {result.status.value.title()}[/red]\n"
                f"[red]Error: {result.error_message or 'Unknown error'}[/red]",
                title=f"Processing Result: {file_path.name}",
                title_align="center"
            )
        console.print(panel)
    else:
        rprint(f"[red]Failed to process {file_path.name}[/red]")


if __name__ == '__main__':
    cli() 