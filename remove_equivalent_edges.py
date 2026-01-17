#!/usr/bin/env python3
"""
Remove EQUIVALENT edges from kg_clean.txt
Removes all triples with 'EQUIVALENT' or 'equivalent' relation
"""

import argparse
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def remove_equivalent_edges(input_path: str, output_path: str = None, backup: bool = True):
    """
    Remove EQUIVALENT edges from knowledge graph file

    Args:
        input_path: Path to kg_clean.txt
        output_path: Path to output file (default: overwrite input)
        backup: Create backup before overwriting (default: True)
    """
    input_file = Path(input_path)

    if not input_file.exists():
        raise FileNotFoundError(f"File not found: {input_path}")

    # Output path
    if output_path is None:
        output_file = input_file
    else:
        output_file = Path(output_path)

    logger.info(f"Reading from: {input_file}")

    # Read all lines
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    total_lines = len(lines)
    logger.info(f"Total lines: {total_lines:,}")

    # Filter out EQUIVALENT edges
    filtered_lines = []
    removed_count = 0

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Parse triple: entity1,relation,entity2
        parts = line.split(',')

        if len(parts) < 3:
            logger.warning(f"Invalid line (skipping): {line[:100]}")
            continue

        # Relation is the middle part
        # Handle cases like: entity1,relation,entity2 or entity1,relation,entity2,extra
        relation = parts[1].strip()

        # Check if relation is EQUIVALENT (case-insensitive)
        if relation.upper() == 'EQUIVALENT':
            removed_count += 1
            logger.debug(f"Removing: {line[:100]}")
        else:
            filtered_lines.append(line + '\n')

    remaining_count = len(filtered_lines)

    logger.info(f"Removed {removed_count:,} EQUIVALENT edges")
    logger.info(f"Remaining lines: {remaining_count:,}")
    logger.info(f"Reduction: {removed_count / total_lines * 100:.2f}%")

    # Create backup if needed
    if backup and output_file == input_file and input_file.exists():
        backup_file = input_file.with_suffix('.txt.bak')
        logger.info(f"Creating backup: {backup_file}")
        import shutil
        shutil.copy2(input_file, backup_file)

    # Write filtered lines
    logger.info(f"Writing to: {output_file}")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(filtered_lines)

    logger.info("✓ Done!")

    # Print statistics
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Input file:       {input_file}")
    print(f"Output file:      {output_file}")
    print(f"Total lines:      {total_lines:,}")
    print(f"Removed:          {removed_count:,} EQUIVALENT edges")
    print(f"Remaining:        {remaining_count:,} edges")
    print(f"Reduction:        {removed_count / total_lines * 100:.2f}%")
    print("=" * 60)

    return remaining_count, removed_count


def main():
    parser = argparse.ArgumentParser(
        description="Remove EQUIVALENT edges from kg_clean.txt"
    )
    parser.add_argument(
        'input_file',
        help='Path to kg_clean.txt'
    )
    parser.add_argument(
        '-o', '--output',
        help='Output file path (default: overwrite input)',
        default=None
    )
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Do not create backup when overwriting'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose logging'
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        remove_equivalent_edges(
            input_path=args.input_file,
            output_path=args.output,
            backup=not args.no_backup
        )
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == '__main__':
    main()
