import hashlib
import json
import os

def hash_file(filepath):
    """
    Compute SHA-256 hash of a file.
    """
    with open(filepath, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def check_or_generate_hash_manifest():
    """
    Check core knowledge files against hash manifest.
    If manifest is missing, create it.
    If any files are missing from manifest, add them.
    If any hashes differ, warn GAIA and flag for re-embedding.
    """
    constants_path = os.path.join("app", "gaia_constants.json")
    manifest_path = os.path.join("knowledge", "hash_manifest.json")

    # Load core knowledge file list
    with open(constants_path, "r") as f:
        constants = json.load(f)
    core_files = constants.get("core_knowledge_files", [])

    # Load or initialize manifest
    if os.path.exists(manifest_path):
        with open(manifest_path, "r") as f:
            hash_manifest = json.load(f)
    else:
        hash_manifest = {}

    updated = False
    changes_detected = False

    for file in core_files:
        file_path = os.path.join("knowledge", file)
        if os.path.exists(file_path):
            file_hash = hash_file(file_path)
            if file not in hash_manifest:
                print(f"📥 Adding new file to hash manifest: {file}")
                hash_manifest[file] = file_hash
                updated = True
            elif hash_manifest[file] != file_hash:
                print(f"⚠️ Hash mismatch detected for {file}")
                hash_manifest[file] = file_hash
                updated = True
                changes_detected = True
        else:
            print(f"⚠️ Core knowledge file missing: {file}")

    if updated:
        with open(manifest_path, "w") as f:
            json.dump(hash_manifest, f, indent=2)
        print("✅ Hash manifest updated.")

    if changes_detected:
        print("⚠️ Knowledge drift detected. Recommend re-embedding GAIA’s knowledge.")
        print("💡 Suggest GAIA propose:")
        print("EXECUTE:\nembed_gaia_reference()")
    else:
        print("✅ Hash manifest is up to date. No re-embedding required.")
