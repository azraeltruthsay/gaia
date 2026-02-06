import os
import json
import hashlib
import shutil
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional, Callable, Any

logger = logging.getLogger("GAIA.SnapshotManager")

class SnapshotManager:
    """
    Tracks file state hashes to detect changes between boots or scan cycles.
    Prevents unnecessary reprocessing of unchanged code.

    Also provides backup/rollback capabilities for safe code modification:
    - backup_file(): Create timestamped backup before editing
    - restore_file(): Rollback to a previous backup
    - safe_edit(): Atomic edit with automatic rollback on failure
    - list_backups(): Show available backups for a file
    - cleanup_old_backups(): Retention management
    """

    # Default retention: keep last 10 backups per file
    DEFAULT_MAX_BACKUPS_PER_FILE = 10

    def __init__(self, config, backup_dir: Optional[str] = None):
        self.config = config
        self.snapshot_path = os.path.join(config.system_reference_path("code_summaries"), "snapshot.json")
        self.current_snapshot = {}
        self.previous_snapshot = self._load_snapshot()

        # Backup directory setup
        if backup_dir:
            self.backup_dir = Path(backup_dir)
        else:
            # Default: knowledge/system_reference/code_backups
            self.backup_dir = Path(config.system_reference_path("code_backups"))

        # Ensure backup directory exists
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"SnapshotManager initialized with backup_dir: {self.backup_dir}")

    def _load_snapshot(self) -> dict:
        if os.path.exists(self.snapshot_path):
            try:
                with open(self.snapshot_path, "r", encoding="utf-8") as f:
                    logger.debug("🗃️ Previous snapshot loaded.")
                    return json.load(f)
            except Exception as e:
                logger.warning(f"⚠️ Failed to load previous snapshot: {e}")
        return {}

    def _hash_file(self, path: str) -> str:
        try:
            with open(path, "rb") as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception:
            return ""

    def update_snapshot(self, file_list: List[str], base_path: str = "/app"):
        """
        Generate new hashes and update current snapshot.
        """
        self.current_snapshot = {
            file: self._hash_file(os.path.join(base_path, file))
            for file in file_list
        }

        try:
            with open(self.snapshot_path, "w", encoding="utf-8") as f:
                json.dump(self.current_snapshot, f, indent=2)
            logger.info("💾 Snapshot updated.")
        except Exception as e:
            logger.error(f"❌ Failed to write snapshot: {e}")

    def get_modified_files(self) -> List[str]:
        """
        Compare current vs previous and return only changed file paths.
        """
        changed = [
            file for file, hash_val in self.current_snapshot.items()
            if self.previous_snapshot.get(file) != hash_val
        ]
        logger.debug(f"🔍 Modified files: {len(changed)}")
        return changed

    # -------------------------------------------------------------------------
    # Backup / Rollback Methods
    # -------------------------------------------------------------------------

    def _get_backup_subdir(self, file_path: str) -> Path:
        """
        Get the backup subdirectory for a given file.
        Preserves relative path structure under backup_dir.
        """
        p = Path(file_path)
        # Use the file's parent path structure as subdirectory
        # e.g., /gaia-assistant/app/cognition/agent_core.py
        #    -> backups/app/cognition/agent_core.py/
        try:
            # Try to make path relative to common roots
            for root in ["/gaia-assistant", "/app", Path.cwd()]:
                try:
                    rel = p.relative_to(root)
                    return self.backup_dir / rel
                except ValueError:
                    continue
            # Fallback: use filename only
            return self.backup_dir / p.name
        except Exception:
            return self.backup_dir / p.name

    def backup_file(self, file_path: str, reason: str = "") -> Dict[str, Any]:
        """
        Create a timestamped backup of a file before modification.

        Args:
            file_path: Absolute or relative path to the file to back up
            reason: Optional reason/description for the backup

        Returns:
            Dict with backup metadata: {ok, backup_path, timestamp, hash, reason, error}
        """
        try:
            src = Path(file_path).resolve()
            if not src.exists():
                return {"ok": False, "error": f"Source file does not exist: {src}"}
            if not src.is_file():
                return {"ok": False, "error": f"Path is not a file: {src}"}

            # Create backup subdirectory
            backup_subdir = self._get_backup_subdir(str(src))
            backup_subdir.mkdir(parents=True, exist_ok=True)

            # Generate timestamp and backup filename
            ts = datetime.now(timezone.utc)
            ts_str = ts.strftime("%Y%m%dT%H%M%SZ")
            backup_name = f"{src.stem}.{ts_str}{src.suffix}"
            backup_path = backup_subdir / backup_name

            # Copy file to backup location
            shutil.copy2(src, backup_path)

            # Calculate hash for verification
            file_hash = self._hash_file(str(src))

            # Write metadata file
            meta_path = backup_path.with_suffix(backup_path.suffix + ".meta.json")
            metadata = {
                "original_path": str(src),
                "backup_path": str(backup_path),
                "timestamp": ts.isoformat(),
                "hash": file_hash,
                "reason": reason,
                "size_bytes": src.stat().st_size,
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Backup created: {backup_path} (reason: {reason or 'none'})")
            return {
                "ok": True,
                "backup_path": str(backup_path),
                "timestamp": ts.isoformat(),
                "hash": file_hash,
                "reason": reason,
            }

        except Exception as e:
            logger.error(f"Failed to backup {file_path}: {e}", exc_info=True)
            return {"ok": False, "error": str(e)}

    def list_backups(self, file_path: str) -> List[Dict[str, Any]]:
        """
        List all available backups for a given file, sorted newest first.

        Args:
            file_path: Path to the original file

        Returns:
            List of backup metadata dicts
        """
        try:
            src = Path(file_path).resolve()
            backup_subdir = self._get_backup_subdir(str(src))

            if not backup_subdir.exists():
                return []

            backups = []
            for meta_file in backup_subdir.glob(f"{src.stem}.*.meta.json"):
                try:
                    with open(meta_file, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                        # Add backup file existence check
                        backup_path = Path(meta.get("backup_path", ""))
                        meta["exists"] = backup_path.exists()
                        backups.append(meta)
                except Exception as e:
                    logger.warning(f"Failed to read backup metadata {meta_file}: {e}")

            # Sort by timestamp, newest first
            backups.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            return backups

        except Exception as e:
            logger.error(f"Failed to list backups for {file_path}: {e}")
            return []

    def restore_file(self, file_path: str, backup_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Restore a file from backup.

        Args:
            file_path: Path to the file to restore
            backup_path: Specific backup to restore from. If None, uses the most recent.

        Returns:
            Dict with restore result: {ok, restored_from, hash_before, hash_after, error}
        """
        try:
            target = Path(file_path).resolve()

            # Get the backup to restore from
            if backup_path:
                src_backup = Path(backup_path)
            else:
                # Use most recent backup
                backups = self.list_backups(str(target))
                if not backups:
                    return {"ok": False, "error": f"No backups found for {target}"}
                src_backup = Path(backups[0]["backup_path"])

            if not src_backup.exists():
                return {"ok": False, "error": f"Backup file not found: {src_backup}"}

            # Record current state before restoring
            hash_before = self._hash_file(str(target)) if target.exists() else None

            # Perform the restore
            shutil.copy2(src_backup, target)

            # Verify restore
            hash_after = self._hash_file(str(target))

            logger.info(f"Restored {target} from {src_backup}")
            return {
                "ok": True,
                "restored_from": str(src_backup),
                "target": str(target),
                "hash_before": hash_before,
                "hash_after": hash_after,
            }

        except Exception as e:
            logger.error(f"Failed to restore {file_path}: {e}", exc_info=True)
            return {"ok": False, "error": str(e)}

    def safe_edit(
        self,
        file_path: str,
        new_content: str,
        reason: str = "",
        validator: Optional[Callable[[str], bool]] = None,
    ) -> Dict[str, Any]:
        """
        Safely edit a file with automatic backup and optional validation.
        Rolls back automatically if validation fails.

        Args:
            file_path: Path to the file to edit
            new_content: New content to write
            reason: Reason for the edit (stored in backup metadata)
            validator: Optional callable that takes file_path and returns True if valid.
                       If validation fails, the file is automatically rolled back.

        Returns:
            Dict with edit result: {ok, backup_path, validated, rolled_back, error}
        """
        target = Path(file_path).resolve()
        backup_result = None
        rolled_back = False

        try:
            # Step 1: Create backup (only if file exists)
            if target.exists():
                backup_result = self.backup_file(str(target), reason=reason)
                if not backup_result.get("ok"):
                    return {
                        "ok": False,
                        "error": f"Backup failed: {backup_result.get('error')}",
                        "stage": "backup",
                    }

            # Step 2: Write new content
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(target, "w", encoding="utf-8") as f:
                f.write(new_content)
            logger.info(f"Wrote {len(new_content)} bytes to {target}")

            # Step 3: Validate if validator provided
            validated = True
            if validator:
                try:
                    validated = validator(str(target))
                    logger.info(f"Validation result for {target}: {validated}")
                except Exception as ve:
                    logger.error(f"Validator raised exception: {ve}")
                    validated = False

            # Step 4: Rollback if validation failed
            if not validated and backup_result:
                logger.warning(f"Validation failed, rolling back {target}")
                restore_result = self.restore_file(str(target), backup_result.get("backup_path"))
                rolled_back = restore_result.get("ok", False)
                return {
                    "ok": False,
                    "error": "Validation failed",
                    "validated": False,
                    "rolled_back": rolled_back,
                    "backup_path": backup_result.get("backup_path"),
                    "stage": "validation",
                }

            return {
                "ok": True,
                "backup_path": backup_result.get("backup_path") if backup_result else None,
                "validated": validated,
                "rolled_back": False,
                "new_hash": self._hash_file(str(target)),
            }

        except Exception as e:
            logger.error(f"safe_edit failed for {file_path}: {e}", exc_info=True)
            # Attempt rollback on any error
            if backup_result and backup_result.get("ok"):
                logger.warning(f"Attempting rollback after error")
                restore_result = self.restore_file(str(target), backup_result.get("backup_path"))
                rolled_back = restore_result.get("ok", False)

            return {
                "ok": False,
                "error": str(e),
                "rolled_back": rolled_back,
                "backup_path": backup_result.get("backup_path") if backup_result else None,
                "stage": "write",
            }

    def cleanup_old_backups(
        self,
        file_path: Optional[str] = None,
        max_backups: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Remove old backups, keeping only the most recent ones.

        Args:
            file_path: If provided, only clean backups for this file.
                       If None, clean all backup directories.
            max_backups: Maximum backups to keep per file. Defaults to DEFAULT_MAX_BACKUPS_PER_FILE.

        Returns:
            Dict with cleanup stats: {ok, removed_count, kept_count, errors}
        """
        max_keep = max_backups or self.DEFAULT_MAX_BACKUPS_PER_FILE
        removed_count = 0
        kept_count = 0
        errors = []

        try:
            if file_path:
                # Clean specific file's backups
                dirs_to_clean = [self._get_backup_subdir(file_path)]
            else:
                # Clean all backup directories
                dirs_to_clean = [d for d in self.backup_dir.rglob("*") if d.is_dir()]
                # Also include the root backup_dir in case files are directly there
                dirs_to_clean.append(self.backup_dir)

            for backup_dir in dirs_to_clean:
                if not backup_dir.exists():
                    continue

                # Group meta files by original filename stem
                meta_files = list(backup_dir.glob("*.meta.json"))
                # Group by base name (without timestamp)
                file_groups: Dict[str, List[Path]] = {}
                for mf in meta_files:
                    # Extract base name: agent_core.20260121T120000Z.py.meta.json -> agent_core
                    parts = mf.name.split(".")
                    if len(parts) >= 4:
                        base = parts[0]
                    else:
                        base = mf.stem
                    file_groups.setdefault(base, []).append(mf)

                for base_name, metas in file_groups.items():
                    # Sort by modification time, newest first
                    metas.sort(key=lambda p: p.stat().st_mtime, reverse=True)

                    # Keep the first max_keep, remove the rest
                    to_keep = metas[:max_keep]
                    to_remove = metas[max_keep:]

                    kept_count += len(to_keep)

                    for meta_path in to_remove:
                        try:
                            # Read meta to find backup file
                            with open(meta_path, "r", encoding="utf-8") as f:
                                meta = json.load(f)
                            backup_file = Path(meta.get("backup_path", ""))

                            # Remove backup file
                            if backup_file.exists():
                                backup_file.unlink()

                            # Remove meta file
                            meta_path.unlink()
                            removed_count += 1
                            logger.debug(f"Removed old backup: {backup_file}")

                        except Exception as e:
                            errors.append(f"Failed to remove {meta_path}: {e}")
                            logger.warning(f"Failed to remove backup {meta_path}: {e}")

            logger.info(f"Backup cleanup complete: kept={kept_count}, removed={removed_count}")
            return {
                "ok": True,
                "removed_count": removed_count,
                "kept_count": kept_count,
                "errors": errors if errors else None,
            }

        except Exception as e:
            logger.error(f"Backup cleanup failed: {e}", exc_info=True)
            return {"ok": False, "error": str(e), "removed_count": removed_count}


# -----------------------------------------------------------------------------
# Built-in Validators for safe_edit()
# -----------------------------------------------------------------------------

def validate_python_syntax(file_path: str) -> bool:
    """
    Validate that a Python file has correct syntax.
    Use with safe_edit() validator parameter.
    """
    import ast
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            source = f.read()
        ast.parse(source)
        logger.info(f"Python syntax valid: {file_path}")
        return True
    except SyntaxError as e:
        logger.error(f"Python syntax error in {file_path}: {e}")
        return False
    except Exception as e:
        logger.error(f"Failed to validate Python syntax for {file_path}: {e}")
        return False


def validate_json_syntax(file_path: str) -> bool:
    """
    Validate that a JSON file has correct syntax.
    Use with safe_edit() validator parameter.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            json.load(f)
        logger.info(f"JSON syntax valid: {file_path}")
        return True
    except json.JSONDecodeError as e:
        logger.error(f"JSON syntax error in {file_path}: {e}")
        return False
    except Exception as e:
        logger.error(f"Failed to validate JSON syntax for {file_path}: {e}")
        return False


def create_import_validator(file_path: str) -> bool:
    """
    Validate that a Python file can be imported without errors.
    More thorough than syntax check but may have side effects.
    Use with safe_edit() validator parameter.
    """
    import importlib.util
    import sys

    try:
        # Create a temporary module spec
        spec = importlib.util.spec_from_file_location("_temp_validation_module", file_path)
        if spec is None or spec.loader is None:
            logger.error(f"Could not create module spec for {file_path}")
            return False

        module = importlib.util.module_from_spec(spec)
        # Don't add to sys.modules to avoid pollution
        try:
            spec.loader.exec_module(module)
            logger.info(f"Import validation passed: {file_path}")
            return True
        except Exception as e:
            logger.error(f"Import validation failed for {file_path}: {e}")
            return False
        finally:
            # Clean up
            if "_temp_validation_module" in sys.modules:
                del sys.modules["_temp_validation_module"]

    except Exception as e:
        logger.error(f"Failed to validate import for {file_path}: {e}")
        return False


def create_pytest_validator(test_pattern: str = "tests/") -> Callable[[str], bool]:
    """
    Create a validator that runs pytest on the specified test directory/pattern.
    Returns a callable suitable for safe_edit() validator parameter.

    Args:
        test_pattern: pytest target (directory, file, or pattern)

    Returns:
        Validator function that returns True if tests pass
    """
    import subprocess

    def validator(file_path: str) -> bool:
        try:
            result = subprocess.run(
                ["python", "-m", "pytest", test_pattern, "-x", "--tb=short", "-q"],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode == 0:
                logger.info(f"pytest validation passed for {file_path}")
                return True
            else:
                logger.error(f"pytest validation failed for {file_path}: {result.stdout}\n{result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            logger.error(f"pytest validation timed out for {file_path}")
            return False
        except Exception as e:
            logger.error(f"pytest validation error for {file_path}: {e}")
            return False

    return validator
