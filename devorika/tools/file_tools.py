"""
File operation tools - Read, Write, Edit files
"""

import os
from pathlib import Path
from typing import Optional, List
from devorika.tools.base import Tool


class ReadFileTool(Tool):
    """Read contents of a file."""

    name = "read_file"
    description = "Read the contents of a file"
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path to the file to read"
            },
            "start_line": {
                "type": "integer",
                "description": "Optional start line number"
            },
            "end_line": {
                "type": "integer",
                "description": "Optional end line number"
            }
        },
        "required": ["file_path"]
    }

    def execute(self, file_path: str, start_line: Optional[int] = None, end_line: Optional[int] = None) -> str:
        """Read file contents."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if start_line is not None or end_line is not None:
                    lines = f.readlines()
                    start = (start_line - 1) if start_line else 0
                    end = end_line if end_line else len(lines)
                    return ''.join(lines[start:end])
                return f.read()
        except Exception as e:
            return f"Error reading file: {str(e)}"


class WriteFileTool(Tool):
    """Write contents to a file."""

    name = "write_file"
    description = "Write content to a file (creates or overwrites)"
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path to the file to write"
            },
            "content": {
                "type": "string",
                "description": "Content to write to the file"
            }
        },
        "required": ["file_path", "content"]
    }

    def execute(self, file_path: str, content: str) -> str:
        """Write content to file."""
        try:
            # Create directory if it doesn't exist
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)

            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return f"Successfully wrote to {file_path}"
        except Exception as e:
            return f"Error writing file: {str(e)}"


class EditFileTool(Tool):
    """Edit a file by replacing specific content."""

    name = "edit_file"
    description = "Edit a file by replacing old content with new content"
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path to the file to edit"
            },
            "old_content": {
                "type": "string",
                "description": "Content to replace"
            },
            "new_content": {
                "type": "string",
                "description": "New content"
            }
        },
        "required": ["file_path", "old_content", "new_content"]
    }

    def execute(self, file_path: str, old_content: str, new_content: str) -> str:
        """Edit file by replacing content."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            if old_content not in content:
                return f"Error: Could not find the specified content in {file_path}"

            new_file_content = content.replace(old_content, new_content)

            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_file_content)

            return f"Successfully edited {file_path}"
        except Exception as e:
            return f"Error editing file: {str(e)}"


class ListDirectoryTool(Tool):
    """List contents of a directory."""

    name = "list_directory"
    description = "List files and directories in a given path"
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Directory path to list"
            },
            "recursive": {
                "type": "boolean",
                "description": "Whether to list recursively"
            }
        },
        "required": ["path"]
    }

    def execute(self, path: str, recursive: bool = False) -> str:
        """List directory contents."""
        try:
            if recursive:
                files = []
                for root, dirs, filenames in os.walk(path):
                    for filename in filenames:
                        files.append(os.path.join(root, filename))
                return "\n".join(files)
            else:
                items = os.listdir(path)
                return "\n".join(items)
        except Exception as e:
            return f"Error listing directory: {str(e)}"


class SearchCodeTool(Tool):
    """Search for code patterns in files."""

    name = "search_code"
    description = "Search for a pattern in code files"
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Search pattern (regex supported)"
            },
            "path": {
                "type": "string",
                "description": "Directory path to search in"
            },
            "file_extension": {
                "type": "string",
                "description": "Optional file extension filter (e.g., '.py')"
            }
        },
        "required": ["pattern", "path"]
    }

    def execute(self, pattern: str, path: str, file_extension: Optional[str] = None) -> str:
        """Search for pattern in files."""
        import re
        results = []

        try:
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file_extension and not file.endswith(file_extension):
                        continue

                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            for i, line in enumerate(f, 1):
                                if re.search(pattern, line):
                                    results.append(f"{file_path}:{i}: {line.strip()}")
                    except:
                        continue

            return "\n".join(results) if results else "No matches found"
        except Exception as e:
            return f"Error searching: {str(e)}"
