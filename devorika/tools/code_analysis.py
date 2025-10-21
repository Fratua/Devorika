"""
Code analysis and understanding tools
"""

import ast
import os
from typing import List, Dict, Any
from devorika.tools.base import Tool


class AnalyzeCodeTool(Tool):
    """Analyze Python code using AST."""

    name = "analyze_code"
    description = "Analyze Python code structure (functions, classes, imports)"
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path to Python file to analyze"
            }
        },
        "required": ["file_path"]
    }

    def execute(self, file_path: str) -> str:
        """Analyze code structure."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()

            tree = ast.parse(code)

            analysis = {
                "imports": [],
                "classes": [],
                "functions": [],
                "globals": []
            }

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        analysis["imports"].append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    for alias in node.names:
                        analysis["imports"].append(f"{module}.{alias.name}")
                elif isinstance(node, ast.ClassDef):
                    bases = [self._get_name(base) for base in node.bases]
                    methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                    analysis["classes"].append({
                        "name": node.name,
                        "bases": bases,
                        "methods": methods,
                        "line": node.lineno
                    })
                elif isinstance(node, ast.FunctionDef) and node.col_offset == 0:
                    args = [arg.arg for arg in node.args.args]
                    analysis["functions"].append({
                        "name": node.name,
                        "args": args,
                        "line": node.lineno
                    })
                elif isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                    if node.col_offset == 0:
                        analysis["globals"].append(node.targets[0].id)

            # Format output
            output = f"Analysis of {file_path}:\n\n"

            if analysis["imports"]:
                output += "Imports:\n"
                for imp in set(analysis["imports"]):
                    output += f"  - {imp}\n"
                output += "\n"

            if analysis["classes"]:
                output += "Classes:\n"
                for cls in analysis["classes"]:
                    bases_str = f"({', '.join(cls['bases'])})" if cls['bases'] else ""
                    output += f"  - {cls['name']}{bases_str} (line {cls['line']})\n"
                    if cls['methods']:
                        output += f"    Methods: {', '.join(cls['methods'])}\n"
                output += "\n"

            if analysis["functions"]:
                output += "Functions:\n"
                for func in analysis["functions"]:
                    args_str = ', '.join(func['args'])
                    output += f"  - {func['name']}({args_str}) (line {func['line']})\n"
                output += "\n"

            return output

        except Exception as e:
            return f"Error analyzing code: {str(e)}"

    def _get_name(self, node):
        """Get name from AST node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        return "?"


class FindBugsTool(Tool):
    """Find potential bugs using static analysis."""

    name = "find_bugs"
    description = "Find potential bugs and code issues using static analysis"
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path to Python file to check"
            }
        },
        "required": ["file_path"]
    }

    def execute(self, file_path: str) -> str:
        """Find bugs using pylint or basic analysis."""
        issues = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()

            # Try using pylint first
            try:
                import subprocess
                result = subprocess.run(
                    ["pylint", file_path, "--output-format=text"],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                return result.stdout
            except:
                pass

            # Fallback to basic AST-based analysis
            tree = ast.parse(code)

            for node in ast.walk(tree):
                # Check for unused variables
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id.startswith('_'):
                            issues.append(f"Line {node.lineno}: Variable '{target.id}' appears unused")

                # Check for bare except
                if isinstance(node, ast.ExceptHandler) and node.type is None:
                    issues.append(f"Line {node.lineno}: Bare except clause (catches all exceptions)")

                # Check for TODO/FIXME comments
            for i, line in enumerate(code.split('\n'), 1):
                if 'TODO' in line or 'FIXME' in line:
                    issues.append(f"Line {i}: {line.strip()}")

            if issues:
                return "Potential issues found:\n" + "\n".join(issues)
            else:
                return "No obvious issues found"

        except Exception as e:
            return f"Error analyzing code: {str(e)}"


class GetCodeComplexityTool(Tool):
    """Calculate code complexity metrics."""

    name = "code_complexity"
    description = "Calculate cyclomatic complexity and other metrics"
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path to Python file"
            }
        },
        "required": ["file_path"]
    }

    def execute(self, file_path: str) -> str:
        """Calculate complexity."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()

            lines = code.split('\n')
            loc = len([l for l in lines if l.strip() and not l.strip().startswith('#')])

            tree = ast.parse(code)
            functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
            classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]

            output = f"Code Metrics for {file_path}:\n"
            output += f"  Lines of Code: {loc}\n"
            output += f"  Functions: {len(functions)}\n"
            output += f"  Classes: {len(classes)}\n"

            return output

        except Exception as e:
            return f"Error calculating complexity: {str(e)}"
