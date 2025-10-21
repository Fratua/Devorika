"""
Code execution and terminal tools
"""

import subprocess
import sys
from typing import Optional, Dict, Any
from devorika.tools.base import Tool


class BashTool(Tool):
    """Execute bash commands."""

    name = "bash"
    description = "Execute bash commands in the terminal"
    parameters = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The bash command to execute"
            },
            "timeout": {
                "type": "integer",
                "description": "Timeout in seconds (default: 30)"
            }
        },
        "required": ["command"]
    }

    def execute(self, command: str, timeout: int = 30) -> str:
        """Execute bash command."""
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout
            )

            output = ""
            if result.stdout:
                output += f"STDOUT:\n{result.stdout}\n"
            if result.stderr:
                output += f"STDERR:\n{result.stderr}\n"
            output += f"Return code: {result.returncode}"

            return output
        except subprocess.TimeoutExpired:
            return f"Command timed out after {timeout} seconds"
        except Exception as e:
            return f"Error executing command: {str(e)}"


class PythonExecuteTool(Tool):
    """Execute Python code."""

    name = "python_execute"
    description = "Execute Python code and return the output"
    parameters = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python code to execute"
            },
            "timeout": {
                "type": "integer",
                "description": "Timeout in seconds (default: 30)"
            }
        },
        "required": ["code"]
    }

    def execute(self, code: str, timeout: int = 30) -> str:
        """Execute Python code."""
        try:
            # Create a temporary file
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name

            # Execute the file
            result = subprocess.run(
                [sys.executable, temp_file],
                capture_output=True,
                text=True,
                timeout=timeout
            )

            # Clean up
            import os
            os.unlink(temp_file)

            output = ""
            if result.stdout:
                output += f"Output:\n{result.stdout}\n"
            if result.stderr:
                output += f"Errors:\n{result.stderr}\n"
            output += f"Return code: {result.returncode}"

            return output
        except subprocess.TimeoutExpired:
            return f"Code execution timed out after {timeout} seconds"
        except Exception as e:
            return f"Error executing code: {str(e)}"


class InstallPackageTool(Tool):
    """Install Python packages."""

    name = "install_package"
    description = "Install a Python package using pip"
    parameters = {
        "type": "object",
        "properties": {
            "package": {
                "type": "string",
                "description": "Package name to install"
            },
            "version": {
                "type": "string",
                "description": "Optional specific version"
            }
        },
        "required": ["package"]
    }

    def execute(self, package: str, version: Optional[str] = None) -> str:
        """Install a package."""
        try:
            pkg = f"{package}=={version}" if version else package
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", pkg],
                capture_output=True,
                text=True,
                timeout=120
            )

            if result.returncode == 0:
                return f"Successfully installed {pkg}"
            else:
                return f"Failed to install {pkg}:\n{result.stderr}"
        except Exception as e:
            return f"Error installing package: {str(e)}"


class RunTestsTool(Tool):
    """Run tests using pytest."""

    name = "run_tests"
    description = "Run tests using pytest"
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to tests (file or directory)"
            },
            "verbose": {
                "type": "boolean",
                "description": "Verbose output"
            }
        },
        "required": ["path"]
    }

    def execute(self, path: str, verbose: bool = True) -> str:
        """Run tests."""
        try:
            args = [sys.executable, "-m", "pytest", path]
            if verbose:
                args.append("-v")

            result = subprocess.run(
                args,
                capture_output=True,
                text=True,
                timeout=300
            )

            output = result.stdout
            if result.stderr:
                output += f"\n\nErrors:\n{result.stderr}"

            return output
        except subprocess.TimeoutExpired:
            return "Tests timed out after 5 minutes"
        except Exception as e:
            return f"Error running tests: {str(e)}"
