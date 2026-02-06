"""
GAIA Rescue Helper
------------------
This file is an extensible utility limb for GAIA.
Its purpose is to provide reloadable, safe-to-edit functions that GAIA can:
- call via `ai.helper.function_name()`
- edit using `ai.read()` + `ai.write()`
- reload dynamically with `ai.reload("utils.gaia_rescue_helper")`

GAIA is encouraged to:
✅ Create helper functions here for log parsing, analysis, file summaries, or sandboxed experiments
✅ Avoid editing core shell files unnecessarily
✅ Think of this file as her personal sandbox limb
"""

import os
import json
import logging
import subprocess  # ✅ Needed for run_shell_safe
import shutil       # ✅ Needed for file backup
import jsonschema   # ✅ Needed for schema validation
from datetime import datetime
from app.config import Config
from app.memory.dev_matrix import GAIADevMatrix

logger = logging.getLogger("GAIA.Helper")


def run_shell_safe(command: str) -> str:
    """Run a shell command safely and return output as string."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=5
        )
        return result.stdout.decode("utf-8").strip()
    except Exception as e:
        return f"❌ Shell error: {e}"

def buffer_and_execute_shell(content: str):
    logger = logging.getLogger("GAIA.ShellExec")
    import re

    match = re.search(r"EXECUTE:\s*(?:```(?:bash|python)?\s*)?(.+?)(?:```)?$", content.strip(), re.DOTALL)
    if not match:
        logger.warning("🛑 EXECUTE block not found or malformed.")
        return

    command = match.group(1).strip()

    # Safety check
    with open("gaia_constants.json", "r") as f:
        safe_funcs = json.load(f).get("safe_execute_functions", [])
    if not any(command.startswith(func) for func in safe_funcs):
        logger.warning(f"⛔ Unsafe command blocked: {command}")
        return

    logger.info(f"🧪 Safe EXECUTE command: {command}")

    from subprocess import Popen, PIPE
    process = Popen(command, shell=True, stdout=PIPE, stderr=PIPE, text=True)
    stdout, stderr = process.communicate()

    from app.memory.status_tracker import GAIAStatus
    result = stdout.strip() if stdout else stderr.strip()
    GAIAStatus.set("last_command_output", result)

    from app.utils.gaia_rescue_helper import sketch
    sketch("ShellCommand", f"EXECUTE: {command}\n\n{result}")

def queue_thought_seed(prompt: str, note: str = "", priority: str = "normal") -> str:
    """
    Save a reflection seed into the official seed file for GAIA to process later.
    """
    seed = {
        "prompt": prompt.strip(),
        "note": note or "Queued from shell",
        "priority": priority.lower(),
        "timestamp": datetime.utcnow().isoformat(),  # Add timestamp for traceability
        "source": "shell"  # Could later allow source customization
    }
    path = "./knowledge/system_reference/thought_seeds/queued_reflections.json"
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # 🔧 Load existing seeds
    seeds = []
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                seeds = json.load(f)
            if not isinstance(seeds, list):
                seeds = []
        except Exception as e:
            logger.warning(f"⚠️ Could not load existing seeds: {e}")
            seeds = []

    # ➕ Append the new seed
    seeds.append(seed)

    # 💾 Save back to file
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(seeds, f, indent=2)
        logger.info(f"🌱 Thought seed queued for reflection: {note or prompt[:60]}")
    except Exception as e:
        logger.error(f"❌ Failed to save thought seeds: {e}")
        return f"❌ Failed to queue thought seed: {e}"

    return f"🌱 Thought seed queued for reflection: {note or prompt[:60]}"

def list_safe_primitives():
    """Return the list of safe shell-callable primitives from GAIA constants."""
    try:
        constants_path = os.path.join(os.path.dirname(__file__), "../gaia_constants.json")
        with open(constants_path, "r", encoding="utf-8") as f:
            constants = json.load(f)
        safe = constants.get("safe_execute_functions", [])
        if not safe:
            return "⚠️ No safe_execute_functions found in constants."
        return "✅ Safe primitives:\n- " + "\n- ".join(safe)
    except Exception as e:
        return f"❌ Failed to load safe primitives: {e}"

def process_thought_seeds():
    """Manually triggers processing of queued thought seeds."""
    from app.cognition.self_reflection import run_self_reflection
    from app.config import Config
    reflection = run_self_reflection(Config())
    reflection.check_for_queued_thoughts()
    return "✅ Processed queued thought seeds."
    
def sketch(title: str, content: str):
    path = "./knowledge/system_reference/sketchpad.json"
    from datetime import datetime
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "title": title,
        "content": content
    }
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = {"sketchpad": []}
        data["sketchpad"].append(entry)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return f"✅ Sketch '{title}' saved."
    except Exception as e:
        return f"❌ Failed to save sketch: {e}"

def show_sketchpad():
    path = "./knowledge/system_reference/sketchpad.json"
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        sketches = data.get("sketchpad", [])
        if not sketches:
            return "📝 Sketchpad is empty."
        output = "\n".join([f"- [{s['timestamp']}] {s['title']}:\n  {s['content']}" for s in sketches])
        return output
    except Exception as e:
        return f"❌ Failed to load sketchpad: {e}"

def clear_sketchpad():
    path = "./knowledge/system_reference/sketchpad.json"
    from datetime import datetime
    try:
        if os.path.exists(path):
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            backup_path = f"{path}.bak.{timestamp}"
            os.rename(path, backup_path)
            logger.info(f"🗃️ Sketchpad backed up to {backup_path}")
        # Create an empty sketchpad after backup
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"sketchpad": []}, f, indent=2)
        return f"🧹 Sketchpad cleared. Backup created at {backup_path}"
    except Exception as e:
        return f"❌ Failed to clear sketchpad: {e}"

def update_json(filepath, key_path, new_value):
    """Update a JSON file at a given nested key_path with safe backup."""
    full_path = os.path.join(Config().KNOWLEDGE_DIR, filepath)
    if not full_path.endswith(".json") or not os.path.exists(full_path):
        print(f"❌ Invalid or missing file: {full_path}")
        return

    with open(full_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Traverse path
    d = data
    for k in key_path[:-1]:
        if k not in d:
            print(f"❌ Path not found: {'.'.join(key_path)}")
            return
        d = d[k]

    d[key_path[-1]] = new_value
    d["_updated"] = datetime.now().isoformat()

     # Optional: Validate against .schema.json
    schema_path = full_path.replace(".json", ".schema.json")
    if os.path.exists(schema_path):
        with open(schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)
        try:
            jsonschema.validate(data, schema)
        except jsonschema.exceptions.ValidationError as e:
            print(f"❌ Schema validation failed: {e.message}")
            return

    # Backup and write
    backup_path = f"{full_path}.{datetime.now().strftime('%Y%m%d%H%M%S')}.bak"
    shutil.copy(full_path, backup_path)

    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"✅ Updated {'.'.join(key_path)} in {filepath}. Backup saved.")

def generate_json_schema(filepath):
    from genson import SchemaBuilder

    full_path = os.path.join(Config.KNOWLEDGE_ROOT, filepath)
    if not os.path.exists(full_path):
        print(f"❌ File not found: {full_path}")
        return

    with open(full_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    builder = SchemaBuilder()
    builder.add_object(data)
    schema = builder.to_schema()

    schema_path = full_path.replace(".json", ".schema.json")
    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)

    print(f"✅ Schema saved: {schema_path}")

def create_codebase_overview(root_path: str = ".", output_file: str = "codebase_overview.md"):
    """
    Runs the 'tree' command to get a codebase overview, filters it,
    and saves it to a markdown file in the artifacts directory for review.
    """
    try:
        # Use a safe, read-only command with ignore patterns
        command = f"tree -a -I '__pycache__|*.pyc|*.gguf|*.db|node_modules|.git'"

        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
            cwd=root_path  # Run command from the project root
        )

        overview_content = f"""# Codebase Overview\n\nGenerated on: {datetime.utcnow().isoformat()}\n```\n{result.stdout.strip()}\n```\n"""
        # We need an instance of Config to get the artifacts directory
        artifacts_dir = Config().ARTIFACTS_DIR
        os.makedirs(artifacts_dir, exist_ok=True)
        output_path = os.path.join(artifacts_dir, output_file)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(overview_content)

        return f"✅ Codebase overview created successfully and saved to {output_path}"

    except FileNotFoundError:
        return "❌ 'tree' command not found. Please install it (`apt-get update && apt-get install -y tree`)."
    except Exception as e:
        return f"❌ Failed to create codebase overview: {e}"