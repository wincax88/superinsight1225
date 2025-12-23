#!/usr/bin/env python3
"""
Script to run Alembic database migrations for SuperInsight Platform.

This script provides a convenient way to run database migrations.
"""

import os
import sys
import subprocess
import logging

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.config.settings import settings

logger = logging.getLogger(__name__)


def run_alembic_command(command_args):
    """
    Run an Alembic command with proper environment setup.
    
    Args:
        command_args (list): List of command arguments for Alembic
        
    Returns:
        bool: True if command succeeded, False otherwise
    """
    try:
        # Change to project root directory
        os.chdir(project_root)
        
        # Prepare the full command
        full_command = [sys.executable, "-m", "alembic"] + command_args
        
        logger.info(f"Running command: {' '.join(full_command)}")
        
        # Run the command
        result = subprocess.run(
            full_command,
            capture_output=True,
            text=True,
            check=True
        )
        
        # Print output
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
            
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Alembic command failed with exit code {e.returncode}")
        if e.stdout:
            print(e.stdout)
        if e.stderr:
            print(e.stderr, file=sys.stderr)
        return False
    except Exception as e:
        logger.error(f"Failed to run Alembic command: {e}")
        return False


def upgrade_database(revision="head"):
    """
    Upgrade database to the specified revision.
    
    Args:
        revision (str): Target revision (default: "head")
        
    Returns:
        bool: True if upgrade succeeded, False otherwise
    """
    logger.info(f"Upgrading database to revision: {revision}")
    return run_alembic_command(["upgrade", revision])


def downgrade_database(revision):
    """
    Downgrade database to the specified revision.
    
    Args:
        revision (str): Target revision
        
    Returns:
        bool: True if downgrade succeeded, False otherwise
    """
    logger.info(f"Downgrading database to revision: {revision}")
    return run_alembic_command(["downgrade", revision])


def show_current_revision():
    """
    Show the current database revision.
    
    Returns:
        bool: True if command succeeded, False otherwise
    """
    logger.info("Showing current database revision")
    return run_alembic_command(["current"])


def show_migration_history():
    """
    Show migration history.
    
    Returns:
        bool: True if command succeeded, False otherwise
    """
    logger.info("Showing migration history")
    return run_alembic_command(["history"])


def create_migration(message, autogenerate=False):
    """
    Create a new migration.
    
    Args:
        message (str): Migration message
        autogenerate (bool): Whether to use autogenerate
        
    Returns:
        bool: True if migration creation succeeded, False otherwise
    """
    logger.info(f"Creating new migration: {message}")
    
    command_args = ["revision", "-m", message]
    if autogenerate:
        command_args.append("--autogenerate")
        
    return run_alembic_command(command_args)


def main():
    """Main function to handle command line arguments."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Alembic database migrations")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Upgrade command
    upgrade_parser = subparsers.add_parser("upgrade", help="Upgrade database")
    upgrade_parser.add_argument("revision", nargs="?", default="head", help="Target revision (default: head)")
    
    # Downgrade command
    downgrade_parser = subparsers.add_parser("downgrade", help="Downgrade database")
    downgrade_parser.add_argument("revision", help="Target revision")
    
    # Current command
    subparsers.add_parser("current", help="Show current revision")
    
    # History command
    subparsers.add_parser("history", help="Show migration history")
    
    # Create migration command
    create_parser = subparsers.add_parser("create", help="Create new migration")
    create_parser.add_argument("message", help="Migration message")
    create_parser.add_argument("--autogenerate", action="store_true", help="Use autogenerate")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Print database configuration
    logger.info(f"Using database: {settings.database.database_url}")
    
    # Execute command
    success = False
    
    if args.command == "upgrade":
        success = upgrade_database(args.revision)
    elif args.command == "downgrade":
        success = downgrade_database(args.revision)
    elif args.command == "current":
        success = show_current_revision()
    elif args.command == "history":
        success = show_migration_history()
    elif args.command == "create":
        success = create_migration(args.message, args.autogenerate)
    else:
        parser.print_help()
        return
    
    if success:
        print("✅ Command completed successfully!")
    else:
        print("❌ Command failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()