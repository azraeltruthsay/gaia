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
    p.add_argument("--model", default="")
    p.add_argument("--port", type=int, default=8092)
    p.add_argument("--device", default="cuda")
    p.add_argument("--compile", default="reduce-overhead",
                   choices=["reduce-overhead", "max-autotune", "none"])
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--managed", action="store_true",
                   help="Start in managed mode — zero GPU, subprocess isolation")
    args = p.parse_args()
    if args.managed:
        from gaia_common.engine import serve_managed
        serve_managed(port=args.port, host=args.host)
    else:
        if not args.model:
            p.error("--model is required in direct mode (or use --managed)")
        serve(args.model, args.port, args.device, args.compile, args.host)
