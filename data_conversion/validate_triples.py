#!/usr/bin/env python3
"""
Validate triples file format and content

Checks:
- File format (comma-separated, 3 columns)
- No empty fields
- Statistics about entities and relations
- Common issues (duplicates, malformed lines)
"""

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

def validate_triples(file_path: str, show_examples: bool = True) -> Dict:
    """Validate triples file and return statistics"""

    path = Path(file_path)
    if not path.exists():
        print(f"❌ Error: File not found: {file_path}")
        return None

    print(f"Validating: {file_path}")
    print("="*60)

    # Statistics
    stats = {
        'total_lines': 0,
        'valid_triples': 0,
        'invalid_lines': [],
        'entities': set(),
        'relations': Counter(),
        'head_entities': set(),
        'tail_entities': set(),
        'duplicates': [],
        'empty_fields': 0,
        'malformed_lines': 0,
    }

    seen_triples = set()
    entity_relations = defaultdict(list)

    # Read and validate
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            stats['total_lines'] += 1
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # Parse triple
            parts = line.split(',')

            # Validate format
            if len(parts) != 3:
                stats['malformed_lines'] += 1
                if len(stats['invalid_lines']) < 10:
                    stats['invalid_lines'].append((line_num, line, f"Expected 3 fields, got {len(parts)}"))
                continue

            head, relation, tail = parts

            # Check for empty fields
            if not head or not relation or not tail:
                stats['empty_fields'] += 1
                if len(stats['invalid_lines']) < 10:
                    stats['invalid_lines'].append((line_num, line, "Empty field detected"))
                continue

            # Valid triple
            stats['valid_triples'] += 1
            stats['entities'].add(head)
            stats['entities'].add(tail)
            stats['relations'][relation] += 1
            stats['head_entities'].add(head)
            stats['tail_entities'].add(tail)

            # Track entity relations
            entity_relations[head].append((relation, tail))
            entity_relations[tail].append((f"^{relation}", head))  # Inverse relation

            # Check for duplicates
            triple = (head, relation, tail)
            if triple in seen_triples:
                stats['duplicates'].append((line_num, line))
            seen_triples.add(triple)

    # Print results
    print(f"\n📊 VALIDATION RESULTS")
    print("="*60)
    print(f"Total lines:          {stats['total_lines']}")
    print(f"Valid triples:        {stats['valid_triples']}")
    print(f"Malformed lines:      {stats['malformed_lines']}")
    print(f"Empty fields:         {stats['empty_fields']}")
    print(f"Duplicates:           {len(stats['duplicates'])}")
    print(f"Unique entities:      {len(stats['entities'])}")
    print(f"Unique relations:     {len(stats['relations'])}")
    print(f"  - Head entities:    {len(stats['head_entities'])}")
    print(f"  - Tail entities:    {len(stats['tail_entities'])}")

    # Entity statistics
    only_head = stats['head_entities'] - stats['tail_entities']
    only_tail = stats['tail_entities'] - stats['head_entities']
    both = stats['head_entities'] & stats['tail_entities']

    print(f"\n📈 ENTITY STATISTICS")
    print("="*60)
    print(f"Entities only as head:     {len(only_head)}")
    print(f"Entities only as tail:     {len(only_tail)}")
    print(f"Entities as both:          {len(both)}")

    # Relation statistics
    print(f"\n🔗 RELATION STATISTICS")
    print("="*60)
    print(f"Total relation types:      {len(stats['relations'])}")
    print(f"\nTop 10 most common relations:")
    for relation, count in stats['relations'].most_common(10):
        print(f"  {relation:30s} {count:6d} triples")

    # Issues
    if stats['invalid_lines']:
        print(f"\n⚠️  INVALID LINES (showing first 10)")
        print("="*60)
        for line_num, line, reason in stats['invalid_lines'][:10]:
            print(f"Line {line_num}: {reason}")
            print(f"  Content: {line[:80]}")

    if stats['duplicates']:
        print(f"\n⚠️  DUPLICATE TRIPLES (showing first 5)")
        print("="*60)
        for line_num, line in stats['duplicates'][:5]:
            print(f"Line {line_num}: {line}")

    # Examples
    if show_examples and stats['valid_triples'] > 0:
        print(f"\n📋 EXAMPLES")
        print("="*60)

        # Sample entities with most relations
        top_entities = sorted(entity_relations.items(),
                             key=lambda x: len(x[1]),
                             reverse=True)[:3]

        print("Entities with most relations:")
        for entity, relations in top_entities:
            print(f"\n  {entity} ({len(relations)} relations):")
            for rel, target in relations[:5]:
                print(f"    - {rel} → {target}")
            if len(relations) > 5:
                print(f"    ... and {len(relations) - 5} more")

    # Final verdict
    print(f"\n{'='*60}")
    if stats['malformed_lines'] == 0 and stats['empty_fields'] == 0:
        print("✅ File is valid! Ready for GFM-RAG pipeline.")
    elif stats['malformed_lines'] > 0 or stats['empty_fields'] > 0:
        print("⚠️  File has issues. Please fix invalid lines before using.")

    if len(stats['duplicates']) > 0:
        print("ℹ️  Note: Duplicates found. Consider running with --deduplicate.")

    print("="*60)

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Validate triples file format and content"
    )
    parser.add_argument('file', help='Triples file to validate')
    parser.add_argument(
        '--no-examples',
        action='store_true',
        help='Hide example triples'
    )

    args = parser.parse_args()

    stats = validate_triples(args.file, show_examples=not args.no_examples)

    if stats is None:
        sys.exit(1)

    # Exit code based on validation
    if stats['malformed_lines'] > 0 or stats['empty_fields'] > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
