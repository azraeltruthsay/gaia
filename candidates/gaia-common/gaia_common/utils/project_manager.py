import os
import json
import logging

logger = logging.getLogger("GAIA.ProjectManager")

class ProjectManager:
    """
    Manages GAIA's project-specific context, including directory mappings,
    persona/instruction association, and dynamic pathing for raw + structured data.
    """

    def __init__(self, config):
        self.config = config
        self.projects_base_path = config.projects_path
        self.active_project = "default"
        self.project_metadata = {}
        self.ensure_default_projects_exist()
        self.set_active_project("default")

    def ensure_default_projects_exist(self):
        """
        Ensures default and dnd-campaign projects exist with sane subdirectory scaffolding.
        """
        for name in ["default", "dnd-campaign"]:
            path = os.path.join(self.projects_base_path, name)
            os.makedirs(path, exist_ok=True)
            for sub in ["raw_data", "structured", "vector_store", "instructions"]:
                os.makedirs(os.path.join(path, sub), exist_ok=True)

    def set_active_project(self, project_name: str):
        """
        Switches the currently active project and loads associated metadata.
        """
        full_path = os.path.join(self.projects_base_path, project_name)
        if not os.path.isdir(full_path):
            logger.warning(f"⚠️ Project '{project_name}' not found. Falling back to 'default'.")
            project_name = "default"
            full_path = os.path.join(self.projects_base_path, project_name)

        self.active_project = project_name
        self.config.project_root_path = full_path

        # Attempt to load project.json
        metadata_path = os.path.join(full_path, "project.json")
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    self.project_metadata = json.load(f)
            except Exception as e:
                logger.warning(f"⚠️ Failed to load metadata for '{project_name}': {e}")
                self.project_metadata = {}
        else:
            self.project_metadata = {}

        logger.info(f"📁 Active project set to: {self.active_project}")

    def get_project_path(self, subdir: str = "") -> str:
        """
        Returns an absolute path for the current project, optionally appended with subdir.
        """
        return os.path.join(self.config.project_root_path, subdir)

    def list_available_projects(self) -> list:
        try:
            return [d for d in os.listdir(self.projects_base_path)
                    if os.path.isdir(os.path.join(self.projects_base_path, d))]
        except Exception as e:
            logger.error(f"❌ Failed to list projects: {e}", exc_info=True)
            return []

    def get_vector_store_path(self) -> str:
        return os.path.join(self.config.project_root_path, "vector_store")

    def get_instruction_file(self) -> str:
        """Returns default instruction path for this project, if present."""
        return os.path.join(self.config.project_root_path, "instructions", "default_instructions.txt")

    def describe(self) -> dict:
        """Returns project name and loaded metadata."""
        return {
            "active_project": self.active_project,
            "metadata": self.project_metadata
        }
