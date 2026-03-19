"""
GAIA Inference Engine — thin wrapper around the shared engine library.

The engine implementation lives in gaia_common.engine (shared across all
tier containers). This module re-exports it for backward compatibility
and adds the CLI entry point.
"""

from gaia_common.engine import serve

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="GAIA Inference Engine")
    p.add_argument("--model", required=True)
    p.add_argument("--port", type=int, default=8092)
    p.add_argument("--device", default="cuda")
    p.add_argument("--compile", default="reduce-overhead",
                   choices=["reduce-overhead", "max-autotune", "none"])
    p.add_argument("--host", default="0.0.0.0")
    args = p.parse_args()
    serve(args.model, args.port, args.device, args.compile, args.host)
