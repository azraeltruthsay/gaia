"""Linguistic Engine — PromptFoo red-teaming runner."""
import asyncio
import json
import logging
import os
import tempfile
from pathlib import Path

log = logging.getLogger("gaia-monkey.linguistic")

PROMPTFOO_SUITES_DIR = Path(__file__).parent.parent / "promptfoo-suites"
CORE_ENDPOINT = os.environ.get("CORE_ENDPOINT", "http://gaia-core:6415")


async def run_suite(suite: str = "persona") -> dict:
    """Run a PromptFoo evaluation suite. Returns structured results."""
    suite_path = PROMPTFOO_SUITES_DIR / f"{suite}.yaml"
    if not suite_path.exists():
        return {"error": f"Suite not found: {suite}", "passed": False}

    output_file = tempfile.mktemp(suffix=".json", prefix="pf_results_")

    try:
        proc = await asyncio.create_subprocess_exec(
            "promptfoo", "eval",
            "--config", str(suite_path),
            "--no-cache",
            "--output", output_file,
            env={**os.environ, "PROMPTFOO_HOME": "/tmp/.promptfoo"},
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=300)

        results_path = Path(output_file)
        if not results_path.exists():
            return {
                "suite": suite,
                "passed": False,
                "error": f"No output file generated. stderr: {stderr.decode()[:500]}",
            }

        raw = json.loads(results_path.read_text())

        # Parse promptfoo output structure
        total = raw.get("results", {}).get("stats", {})
        passes = total.get("successes", 0)
        failures = total.get("failures", 0)
        total_count = passes + failures

        failed_assertions = []
        for test in raw.get("results", {}).get("results", []):
            if not test.get("success"):
                failed_assertions.append({
                    "prompt": str(test.get("prompt", ""))[:100],
                    "output": str(test.get("response", {}).get("output", ""))[:200],
                    "reason": str(test.get("failureReason", ""))[:200],
                })

        return {
            "suite": suite,
            "passed": failures == 0,
            "passes": passes,
            "failures": failures,
            "total": total_count,
            "failed_assertions": failed_assertions[:5],
        }

    except asyncio.TimeoutError:
        return {"suite": suite, "passed": False, "error": "PromptFoo timed out after 300s"}
    except FileNotFoundError:
        return {"suite": suite, "passed": False, "error": "promptfoo not found in PATH"}
    except Exception as e:
        return {"suite": suite, "passed": False, "error": str(e)[:300]}
    finally:
        try:
            Path(output_file).unlink(missing_ok=True)
        except Exception:
            pass
