import asyncio
import os
import sys
from notebooklm import NotebookLMClient

NOTEBOOK_NAME = "GAIA Codebase"
STORAGE_STATE = "/home/azrael/.notebooklm/storage_state.json"

async def cleanup():
    client = await NotebookLMClient.from_storage(path=STORAGE_STATE)
    async with client:
        notebooks = await client.notebooks.list()
        target = next((nb for nb in notebooks if nb.title == NOTEBOOK_NAME), None)
        if not target:
            print(f"Notebook '{NOTEBOOK_NAME}' not found.")
            return

        print(f"Cleaning up '{NOTEBOOK_NAME}'...")
        sources = await client.sources.list(target.id)
        
        to_delete = [
            s for s in sources 
            if ".beads" in s.title 
            or s.title.startswith("data_") 
            or s.title.startswith("scripts_")
            or s.title in [
                "count_file_types.sh.txt",
                "create_flat_representation.sh.txt",
                "interactive_gaia_start.sh.txt",
                "notebooklm_sync.py.txt",
                "patch_scripts.py.txt",
                "patch_world_state_read_only.py.txt",
                "run_dnd_test.sh.txt",
                "start_notebooklm_sync.sh.txt",
                "test_gaia_core_imports.sh.txt"
            ]
        ]
        
        if not to_delete:
            print("No .beads sources found.")
            return

        print(f"Found {len(to_delete)} .beads sources to remove.")
        for s in to_delete:
            print(f"Deleting: {s.title}...", end=" ", flush=True)
            try:
                await client.sources.delete(target.id, s.id)
                print("OK")
            except Exception as e:
                print(f"FAILED: {e}")
            await asyncio.sleep(0.5)

if __name__ == "__main__":
    asyncio.run(cleanup())
