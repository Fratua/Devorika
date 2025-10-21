"""
Git version control tools
"""

import subprocess
from typing import Optional
from devorika.tools.base import Tool


class GitStatusTool(Tool):
    """Get git repository status."""

    name = "git_status"
    description = "Get the current git status"
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Repository path (default: current directory)"
            }
        }
    }

    def execute(self, path: str = ".") -> str:
        """Get git status."""
        try:
            result = subprocess.run(
                ["git", "-C", path, "status"],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.stdout if result.returncode == 0 else result.stderr
        except Exception as e:
            return f"Error getting git status: {str(e)}"


class GitDiffTool(Tool):
    """Get git diff."""

    name = "git_diff"
    description = "Get git diff of changes"
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Repository path (default: current directory)"
            },
            "file": {
                "type": "string",
                "description": "Specific file to diff"
            }
        }
    }

    def execute(self, path: str = ".", file: Optional[str] = None) -> str:
        """Get git diff."""
        try:
            cmd = ["git", "-C", path, "diff"]
            if file:
                cmd.append(file)

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.stdout if result.stdout else "No changes"
        except Exception as e:
            return f"Error getting git diff: {str(e)}"


class GitCommitTool(Tool):
    """Create a git commit."""

    name = "git_commit"
    description = "Create a git commit with a message"
    parameters = {
        "type": "object",
        "properties": {
            "message": {
                "type": "string",
                "description": "Commit message"
            },
            "path": {
                "type": "string",
                "description": "Repository path (default: current directory)"
            },
            "add_all": {
                "type": "boolean",
                "description": "Add all changes before committing"
            }
        },
        "required": ["message"]
    }

    def execute(self, message: str, path: str = ".", add_all: bool = True) -> str:
        """Create git commit."""
        try:
            if add_all:
                subprocess.run(
                    ["git", "-C", path, "add", "-A"],
                    capture_output=True,
                    timeout=10
                )

            result = subprocess.run(
                ["git", "-C", path, "commit", "-m", message],
                capture_output=True,
                text=True,
                timeout=10
            )

            return result.stdout if result.returncode == 0 else result.stderr
        except Exception as e:
            return f"Error creating commit: {str(e)}"


class GitLogTool(Tool):
    """Get git log."""

    name = "git_log"
    description = "Get git commit history"
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Repository path (default: current directory)"
            },
            "limit": {
                "type": "integer",
                "description": "Number of commits to show (default: 10)"
            }
        }
    }

    def execute(self, path: str = ".", limit: int = 10) -> str:
        """Get git log."""
        try:
            result = subprocess.run(
                ["git", "-C", path, "log", f"-{limit}", "--oneline"],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.stdout if result.returncode == 0 else result.stderr
        except Exception as e:
            return f"Error getting git log: {str(e)}"


class GitBranchTool(Tool):
    """Manage git branches."""

    name = "git_branch"
    description = "List, create, or switch git branches"
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "description": "Action: 'list', 'create', or 'switch'"
            },
            "branch_name": {
                "type": "string",
                "description": "Branch name (for create/switch)"
            },
            "path": {
                "type": "string",
                "description": "Repository path (default: current directory)"
            }
        },
        "required": ["action"]
    }

    def execute(self, action: str, branch_name: Optional[str] = None, path: str = ".") -> str:
        """Manage git branches."""
        try:
            if action == "list":
                result = subprocess.run(
                    ["git", "-C", path, "branch", "-a"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
            elif action == "create" and branch_name:
                result = subprocess.run(
                    ["git", "-C", path, "checkout", "-b", branch_name],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
            elif action == "switch" and branch_name:
                result = subprocess.run(
                    ["git", "-C", path, "checkout", branch_name],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
            else:
                return "Invalid action or missing branch_name"

            return result.stdout if result.returncode == 0 else result.stderr
        except Exception as e:
            return f"Error managing branches: {str(e)}"


class GitPushTool(Tool):
    """Push changes to remote."""

    name = "git_push"
    description = "Push commits to remote repository"
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Repository path (default: current directory)"
            },
            "remote": {
                "type": "string",
                "description": "Remote name (default: origin)"
            },
            "branch": {
                "type": "string",
                "description": "Branch name (default: current branch)"
            }
        }
    }

    def execute(self, path: str = ".", remote: str = "origin", branch: Optional[str] = None) -> str:
        """Push to remote."""
        try:
            cmd = ["git", "-C", path, "push", remote]
            if branch:
                cmd.append(branch)

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )

            return result.stdout if result.returncode == 0 else result.stderr
        except Exception as e:
            return f"Error pushing to remote: {str(e)}"
