#!/usr/bin/env python3
"""
AAAK Tool — command-line utility to compress/decompress using the AAAK dialect.
"""

import sys
from pathlib import Path

# Add gaia-common to path to import AAKDialect
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "gaia-common"))

try:
    from gaia_common.utils.aaak_dialect import AAKDialect
except ImportError:
    # Fallback to candidates folder if live is not set up
    sys.path.insert(0, str(PROJECT_ROOT / "candidates" / "gaia-common"))
    from gaia_common.utils.aaak_dialect import AAKDialect

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Compress or decompress text using AAAK dialect.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--compress", "-c", action="store_true", help="Compress text from input file or stdin")
    group.add_argument("--decompress", "-d", action="store_true", help="Decompress/summarize AAAK text")
    parser.add_argument("--file", "-f", help="Input file path (defaults to stdin)")
    parser.add_argument("--out", "-o", help="Output file path (defaults to stdout)")
    parser.add_argument("--wing", help="Metadata: wing name")
    parser.add_argument("--room", help="Metadata: room name")
    parser.add_argument("--source", help="Metadata: source name")

    args = parser.parse_args()

    # Read input
    if args.file:
        text = Path(args.file).read_text(encoding="utf-8")
    else:
        text = sys.stdin.read()

    dialect = AAKDialect()

    if args.compress:
        metadata = {
            "wing": args.wing or "",
            "room": args.room or "",
            "source": args.source or "",
        }
        result = dialect.compress(text, metadata=metadata)
    else:
        # AAAK is lossy compression/abbreviation, so decompress displays the structured fields
        lines = text.strip().split("\n")
        out_lines = []
        for line in lines:
            if not line.strip():
                continue
            if "|" in line:
                parts = line.split("|")
                if len(parts) == 4 and not parts[0].startswith("0:"):
                    out_lines.append(f"Location: {parts[0]}/{parts[1]} | Date: {parts[2]} | Source: {parts[3]}")
                elif parts[0].startswith("0:"):
                    entities = parts[0][2:]
                    topics = parts[1] if len(parts) > 1 else ""
                    quote = parts[2] if len(parts) > 2 else ""
                    emotion = parts[3] if len(parts) > 3 else ""
                    flags = parts[4] if len(parts) > 4 else ""
                    out_lines.append(f"  Entities: {entities}")
                    out_lines.append(f"  Topics:   {topics}")
                    out_lines.append(f"  Summary:  {quote}")
                    if emotion:
                        out_lines.append(f"  Affect:   {emotion}")
                    if flags:
                        out_lines.append(f"  Flags:    {flags}")
            else:
                out_lines.append(line)
        result = "\n".join(out_lines)

    # Write output
    if args.out:
        Path(args.out).write_text(result, encoding="utf-8")
        print(f"Output written to {args.out}")
    else:
        print(result)

if __name__ == "__main__":
    main()
