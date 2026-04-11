import asyncio
import os
import sys
import traceback
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from notebooklm import NotebookLMClient, SourceStatus, SourceProcessingError, SourceTimeoutError, SourceNotFoundError, SourceAddError
from notebooklm.paths import get_storage_path, get_browser_profile_dir
from typing import List, Optional

# Configuration
WATCH_DIRECTORY = "/gaia/gaia-instance/artifacts/GAIA_Condensed_flat"
NOTEBOOK_NAME = "GAIA Codebase"
UPLOAD_WAIT_TIMEOUT = 180.0  # seconds to wait for source to become READY (raised from 90)
MAX_RETRIES = 3               # retry failed uploads this many times (raised from 2)
VALIDATION_INTERVAL = 600     # seconds between periodic validation sweeps (10 min, was 5)
AUTH_REFRESH_COOLDOWN = 600   # min seconds between auto-refresh attempts (10 min)
UPLOAD_RATE_LIMIT = 3.0       # seconds between consecutive uploads (prevent API flood)
MAX_FILE_SIZE_BYTES = 500_000 # skip files larger than 500KB (Google often times out on these)

_last_auth_refresh = 0.0


def _refresh_auth_headless() -> bool:
    """Re-export storage_state.json from the persistent browser profile.

    The persistent profile at ~/.notebooklm/browser_profile/ maintains
    Google's session across browser launches (Chrome handles cookie refresh
    internally). We launch headless Chromium with this profile, navigate to
    NotebookLM, and re-export the cookies.

    Returns True if refresh succeeded, False otherwise.
    """
    import time
    global _last_auth_refresh
    now = time.time()
    if now - _last_auth_refresh < AUTH_REFRESH_COOLDOWN:
        remaining = int(AUTH_REFRESH_COOLDOWN - (now - _last_auth_refresh))
        print(f"Auth refresh on cooldown ({remaining}s remaining). Skipping.")
        return False
    _last_auth_refresh = now

    storage_path = get_storage_path()
    browser_profile = get_browser_profile_dir()

    if not browser_profile.exists():
        print("No persistent browser profile found. Manual 'notebooklm login' required.")
        return False

    print("Auto-refreshing auth from persistent browser profile (headless)...")
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            context = p.chromium.launch_persistent_context(
                user_data_dir=str(browser_profile),
                headless=True,
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--password-store=basic",
                ],
                ignore_default_args=["--enable-automation"],
            )
            page = context.pages[0] if context.pages else context.new_page()
            page.goto(
                "https://notebooklm.google.com/",
                wait_until="domcontentloaded",
                timeout=30000,
            )
            # Check if we actually landed on NotebookLM (not a login redirect)
            final_url = page.url
            if "accounts.google.com" in final_url:
                print(f"Auto-refresh failed: redirected to login ({final_url[:80]}...)")
                print("Manual 'notebooklm login' required to re-authenticate.")
                context.close()
                return False

            context.storage_state(path=str(storage_path))
            storage_path.chmod(0o600)
            context.close()
            print(f"Auth refreshed successfully. Cookies saved to {storage_path}")
            return True
    except ImportError:
        print("Playwright not installed. Cannot auto-refresh auth.")
        return False
    except Exception as e:
        print(f"Auth refresh failed: {e}")
        return False


async def validate_sources(client, notebook_id) -> int:
    """Check all remote sources for ERROR state and remove them.
    Returns the count of sources removed."""
    try:
        remote_sources = await client.sources.list(notebook_id)
        if not isinstance(remote_sources, list):
            return 0
    except Exception as e:
        print(f"ERROR: validation could not list sources: {e}")
        return 0

    removed = 0
    for source in remote_sources:
        if source.is_error:
            print(f"VALIDATION: Source '{source.title}' (id={source.id}) is in ERROR state. Removing...")
            try:
                await client.sources.delete(notebook_id, source.id)
                removed += 1
                await asyncio.sleep(0.2)
            except Exception as e:
                print(f"  Failed to delete errored source '{source.title}': {e}")
        elif source.is_processing:
            # Check if it's been stuck processing — try wait_until_ready with short timeout
            try:
                await client.sources.wait_until_ready(notebook_id, source.id, timeout=15.0)
            except SourceTimeoutError:
                print(f"VALIDATION: Source '{source.title}' stuck PROCESSING. Removing...")
                try:
                    await client.sources.delete(notebook_id, source.id)
                    removed += 1
                except Exception:
                    pass
            except (SourceProcessingError, SourceNotFoundError):
                print(f"VALIDATION: Source '{source.title}' failed during wait. Removing...")
                try:
                    await client.sources.delete(notebook_id, source.id)
                    removed += 1
                except Exception:
                    pass
            except Exception:
                pass

    if removed > 0:
        print(f"VALIDATION: Removed {removed} failed source(s).")
    return removed


async def upload_with_validation(client, notebook_id, file_name, file_path, retries=MAX_RETRIES) -> bool:
    """Upload a file and wait for it to reach READY state. Retry with exponential backoff."""
    # Pre-flight: skip oversized files (Google often times out on large uploads)
    try:
        file_size = os.path.getsize(file_path)
        if file_size > MAX_FILE_SIZE_BYTES:
            print(f"  SKIP: {file_name} ({file_size // 1024}KB > {MAX_FILE_SIZE_BYTES // 1024}KB limit)")
            return False
        if file_size == 0:
            print(f"  SKIP: {file_name} (empty file)")
            return False
    except OSError:
        return False

    for attempt in range(1, retries + 1):
        try:
            print(f"  -> {file_name} (attempt {attempt}/{retries}, {file_size // 1024}KB)")
            source = await client.sources.add_file(notebook_id, file_path)

            # Wait for processing to complete (extended timeout)
            ready_source = await client.sources.wait_until_ready(
                notebook_id, source.id, timeout=UPLOAD_WAIT_TIMEOUT
            )
            if ready_source.is_ready:
                print(f"  OK: {file_name}")
                return True

        except SourceAddError as e:
            print(f"  UPLOAD FAILED: {file_name} — {e}")
        except SourceTimeoutError as e:
            print(f"  TIMEOUT: {file_name} did not become ready within {e.timeout}s (last status: {e.last_status})")
            try:
                await client.sources.delete(notebook_id, e.source_id)
                print(f"  Cleaned up stuck source for {file_name}")
            except Exception:
                pass
        except SourceProcessingError as e:
            print(f"  PROCESSING ERROR: {file_name} — status {e.status}")
            try:
                await client.sources.delete(notebook_id, e.source_id)
                print(f"  Cleaned up errored source for {file_name}")
            except Exception:
                pass
        except SourceNotFoundError:
            print(f"  NOT FOUND: {file_name} source disappeared during processing")
        except Exception as e:
            err_str = str(e)
            if "timed out" in err_str.lower() or "timeout" in err_str.lower():
                print(f"  RPC TIMEOUT: {file_name} — {err_str[:80]}")
            else:
                print(f"  UNEXPECTED ERROR uploading {file_name}: {err_str[:120]}")

        if attempt < retries:
            # Exponential backoff: 4s, 8s, 16s (give Google API breathing room)
            wait = 4 * (2 ** (attempt - 1))
            print(f"  Retrying {file_name} in {wait}s...")
            await asyncio.sleep(wait)

    print(f"  GAVE UP: {file_name} after {retries} attempts")
    return False


async def reconcile_state(client, notebook_id, modified_file_name: Optional[str] = None):
    print("Starting State Reconciliation...")

    # 1. Get Remote State
    remote_sources = []
    try:
        raw_remote_sources = await client.sources.list(notebook_id)
        if raw_remote_sources is None:
            print("WARNING: client.sources.list returned None.")
            remote_sources = []
        elif not isinstance(raw_remote_sources, list):
            print(f"WARNING: client.sources.list returned {type(raw_remote_sources)}, expected list.")
            remote_sources = []
        else:
            remote_sources = raw_remote_sources
            print(f"Found {len(remote_sources)} remote sources.")
    except Exception as e:
        print(f"ERROR: Failed to list remote sources: {e}")
        traceback.print_exc(file=sys.stdout)
        remote_sources = []

    # Filter out sources already in ERROR state during reconciliation
    errored = [s for s in remote_sources if s.is_error]
    if errored:
        print(f"Found {len(errored)} source(s) in ERROR state — removing before reconciliation...")
        for s in errored:
            try:
                await client.sources.delete(notebook_id, s.id)
                await asyncio.sleep(0.2)
                print(f"  Removed errored: {s.title}")
            except Exception as e:
                print(f"  Failed to remove errored source {s.title}: {e}")
        # Re-list after cleanup
        remote_sources = [s for s in remote_sources if not s.is_error]

    # Build remote map and detect duplicates
    # A dict overwrites earlier entries, so duplicates become invisible.
    # We need to find all source IDs per title, keep the newest, delete the rest.
    remote_by_title: dict[str, list] = {}
    for s in remote_sources:
        remote_by_title.setdefault(s.title, []).append(s)

    duplicate_ids = []
    for title, sources in remote_by_title.items():
        if len(sources) > 1:
            # Keep the most recent (last in list), delete the rest
            sources.sort(key=lambda x: x.created_at or 0)
            duplicate_ids.extend(s.id for s in sources[:-1])
            print(f"DEDUP: '{title}' has {len(sources)} copies — removing {len(sources) - 1}")

    if duplicate_ids:
        print(f"Removing {len(duplicate_ids)} duplicate source(s)...")
        for source_id in duplicate_ids:
            try:
                await client.sources.delete(notebook_id, source_id)
                await asyncio.sleep(0.2)
            except Exception as e:
                print(f"  Failed to delete duplicate {source_id}: {e}")

    # After dedup, build a clean 1:1 map
    remote_map = {}
    for title, sources in remote_by_title.items():
        # Pick the one we kept (last after sort), or first if only one
        kept = [s for s in sources if s.id not in duplicate_ids]
        if kept:
            remote_map[title] = kept[0].id

    # 2. Get Local State
    local_files = [f for f in os.listdir(WATCH_DIRECTORY) if os.path.isfile(os.path.join(WATCH_DIRECTORY, f))]
    local_set = set(local_files)

    to_delete = []
    to_upload = []

    if modified_file_name:
        print(f"Detected modification for: {modified_file_name}")
        if modified_file_name in remote_map:
            to_delete.append(remote_map[modified_file_name])
        if modified_file_name in local_set:
            to_upload.append(modified_file_name)
        else:
            print(f"Modified file {modified_file_name} not found locally. Treating as deletion.")
            if modified_file_name in remote_map:
                to_delete.append(remote_map[modified_file_name])
    else:
        to_delete = [remote_map[name] for name in remote_map if name not in local_set]
        to_upload = [name for name in local_set if name not in remote_map]

    # 3. Execute Deletions
    if to_delete:
        print(f"Removing {len(to_delete)} orphaned/stale sources...")
        for source_id in to_delete:
            try:
                await client.sources.delete(notebook_id, source_id)
            except Exception as e:
                print(f"  Delete failed for {source_id}: {e}")
            await asyncio.sleep(0.2)

    # 4. Execute Uploads with validation
    if to_upload:
        print(f"Uploading {len(to_upload)} new/updated sources...")
        succeeded = 0
        failed = 0
        for file_name in to_upload:
            file_path = os.path.join(WATCH_DIRECTORY, file_name)
            if not os.path.exists(file_path):
                print(f"  File {file_name} disappeared before upload. Skipping.")
                continue
            if os.path.getsize(file_path) == 0:
                print(f"  Skipping empty file: {file_name}")
                continue
            ok = await upload_with_validation(client, notebook_id, file_name, file_path)
            if ok:
                succeeded += 1
            else:
                failed += 1
            # Rate limit: prevent API flood (was 0.3s — now configurable)
            await asyncio.sleep(UPLOAD_RATE_LIMIT)

        print(f"Upload results: {succeeded} succeeded, {failed} failed")

    if not to_delete and not to_upload:
        if not modified_file_name:
            print("Local and remote states are already in sync.")
        else:
            print(f"Modified file {modified_file_name} processed.")
    else:
        print("Synchronization complete.")


class SyncHandler(FileSystemEventHandler):
    """Collects file events into a batch, then runs ONE reconciliation after a
    quiet period.  This prevents the N-concurrent-reconciliation storm that
    occurs when flatten_soa.sh touches dozens of files at once."""

    BATCH_WINDOW = 5.0  # seconds of quiet before flushing the batch

    def __init__(self, loop, notebook_id):
        self.loop = loop
        self.notebook_id = notebook_id
        self._pending: set = set()
        self._flush_handle: Optional[asyncio.TimerHandle] = None
        self._lock = asyncio.Lock()

    def _schedule_flush(self):
        """(Re)start the batch timer.  Called from watchdog thread."""
        if self._flush_handle is not None:
            self._flush_handle.cancel()
        self._flush_handle = self.loop.call_later(
            self.BATCH_WINDOW, lambda: asyncio.ensure_future(self._flush_batch())
        )

    def _on_event(self, file_name):
        self._pending.add(file_name)
        self.loop.call_soon_threadsafe(self._schedule_flush)

    async def _flush_batch(self):
        async with self._lock:
            batch = self._pending.copy()
            self._pending.clear()
        if not batch:
            return
        print(f"Batch sync: {len(batch)} file(s) changed, running single reconciliation...")
        try:
            async with await NotebookLMClient.from_storage() as client:
                # Per-file reconcile for each changed file (sequentially, one client)
                for file_name in sorted(batch):
                    await reconcile_state(client, self.notebook_id, modified_file_name=file_name)
        except Exception as e:
            print(f"Batch sync error: {e}")

    def on_modified(self, event):
        if not event.is_directory:
            self._on_event(os.path.basename(event.src_path))

    def on_created(self, event):
        if not event.is_directory:
            self._on_event(os.path.basename(event.src_path))

    def on_deleted(self, event):
        if not event.is_directory:
            self._on_event(os.path.basename(event.src_path))


async def periodic_validation(client, notebook_id):
    """Background task: periodically validate sources and do a full reconcile
    to catch any drift from missed watchdog events."""
    while True:
        await asyncio.sleep(VALIDATION_INTERVAL)
        print(f"--- Periodic validation + reconciliation sweep ---")
        try:
            async with await NotebookLMClient.from_storage() as fresh_client:
                await validate_sources(fresh_client, notebook_id)
                # Always do a full reconcile to catch drift (deleted/added files
                # that watchdog may have missed after long uptime)
                await reconcile_state(fresh_client, notebook_id)
        except ValueError as e:
            err_msg = str(e)
            if "expired" in err_msg.lower() or "login" in err_msg.lower() or "redirect" in err_msg.lower():
                print(f"Periodic validation: auth expired. Attempting auto-refresh...")
                loop = asyncio.get_running_loop()
                refreshed = await loop.run_in_executor(None, _refresh_auth_headless)
                if refreshed:
                    print("Auth refreshed. Next validation sweep will use new cookies.")
                else:
                    print("Auto-refresh failed. Run 'notebooklm login' manually.")
            else:
                print(f"Periodic validation error: {e}")
        except Exception as e:
            print(f"Periodic validation error: {e}")


async def start_watcher():
    print("--- start_watcher() called ---")
    if not os.path.exists(WATCH_DIRECTORY):
        print(f"Creating WATCH_DIRECTORY: {WATCH_DIRECTORY}")
        os.makedirs(WATCH_DIRECTORY)

    # Try connecting; auto-refresh auth if expired
    max_auth_retries = 2
    for auth_attempt in range(max_auth_retries + 1):
        try:
            print("Connecting to NotebookLM...")
            client_ctx = await NotebookLMClient.from_storage()
            break
        except ValueError as e:
            err_msg = str(e)
            if ("expired" in err_msg.lower() or "login" in err_msg.lower() or "redirect" in err_msg.lower()) and auth_attempt < max_auth_retries:
                print(f"Auth expired on startup (attempt {auth_attempt + 1}/{max_auth_retries}). Auto-refreshing...")
                loop = asyncio.get_running_loop()
                refreshed = await loop.run_in_executor(None, _refresh_auth_headless)
                if not refreshed:
                    print("Auto-refresh failed. Run 'notebooklm login' manually.")
                    sys.exit(1)
                continue
            raise

    try:
        async with client_ctx as client:
            notebooks = await client.notebooks.list()
            target_nb = next((nb for nb in notebooks if nb.title == NOTEBOOK_NAME), None)

            if not target_nb:
                print(f"Notebook '{NOTEBOOK_NAME}' not found. Creating new notebook.")
                target_nb = await client.notebooks.create(NOTEBOOK_NAME)
            else:
                print(f"Notebook '{NOTEBOOK_NAME}' found (ID: {target_nb.id}).")

            # Validate existing sources before initial reconciliation
            print("Running pre-sync validation...")
            await validate_sources(client, target_nb.id)

            print("Performing initial state reconciliation...")
            await reconcile_state(client, target_nb.id)

            print(f"Monitoring {WATCH_DIRECTORY} for changes. Press Ctrl+C to stop.")

            loop = asyncio.get_running_loop()
            handler = SyncHandler(loop, target_nb.id)
            observer = Observer()
            observer.schedule(handler, WATCH_DIRECTORY, recursive=False)
            observer.start()
            print("Watchdog observer started.")

            # Start periodic validation in background
            validation_task = asyncio.create_task(periodic_validation(client, target_nb.id))

            try:
                while True:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                print("KeyboardInterrupt detected. Stopping...")
                observer.stop()
                validation_task.cancel()
            finally:
                observer.join()
                print("Observer joined.")
    except Exception as e:
        print(f"FATAL ERROR in start_watcher(): {e}", file=sys.stdout, flush=True)
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)


if __name__ == "__main__":
    print("--- NotebookLM Sync with validation ---")
    try:
        asyncio.run(start_watcher())
    except Exception as e:
        print(f"FATAL ERROR during asyncio.run(): {e}", file=sys.stdout, flush=True)
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)
