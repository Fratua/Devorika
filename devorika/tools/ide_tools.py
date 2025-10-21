"""
IDE Integration Tools for Devorika
Provides advanced IDE capabilities including code intelligence, refactoring, and navigation.
"""

import os
import ast
import re
import json
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import subprocess
from .base import Tool


class CodeIntelligenceTool(Tool):
    """
    Advanced code intelligence tool using AST and static analysis.
    Provides symbol definitions, references, and type inference.
    """

    name = "code_intelligence"
    description = "Get symbol definitions, references, type information, and code structure"

    def execute(self, file_path: str, symbol: Optional[str] = None,
                action: str = "definitions") -> Dict[str, Any]:
        """
        Execute code intelligence analysis.

        Args:
            file_path: Path to Python file
            symbol: Symbol name to analyze (optional)
            action: Action to perform (definitions, references, symbols, outline)

        Returns:
            Dict containing analysis results
        """
        try:
            if not os.path.exists(file_path):
                return {"error": f"File not found: {file_path}"}

            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()

            tree = ast.parse(source_code, filename=file_path)

            if action == "definitions" and symbol:
                return self._find_definitions(tree, symbol, source_code)
            elif action == "references" and symbol:
                return self._find_references(tree, symbol, source_code)
            elif action == "symbols":
                return self._extract_all_symbols(tree, source_code)
            elif action == "outline":
                return self._generate_outline(tree, source_code)
            else:
                return {"error": f"Invalid action: {action}"}

        except SyntaxError as e:
            return {"error": f"Syntax error in file: {str(e)}"}
        except Exception as e:
            return {"error": f"Code intelligence failed: {str(e)}"}

    def _find_definitions(self, tree: ast.AST, symbol: str, source: str) -> Dict[str, Any]:
        """Find all definitions of a symbol."""
        definitions = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == symbol:
                definitions.append({
                    'type': 'function',
                    'name': symbol,
                    'line': node.lineno,
                    'col': node.col_offset,
                    'docstring': ast.get_docstring(node),
                    'args': [arg.arg for arg in node.args.args]
                })
            elif isinstance(node, ast.ClassDef) and node.name == symbol:
                definitions.append({
                    'type': 'class',
                    'name': symbol,
                    'line': node.lineno,
                    'col': node.col_offset,
                    'docstring': ast.get_docstring(node),
                    'methods': [m.name for m in node.body if isinstance(m, ast.FunctionDef)]
                })
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == symbol:
                        definitions.append({
                            'type': 'variable',
                            'name': symbol,
                            'line': node.lineno,
                            'col': node.col_offset
                        })

        return {'symbol': symbol, 'definitions': definitions}

    def _find_references(self, tree: ast.AST, symbol: str, source: str) -> Dict[str, Any]:
        """Find all references to a symbol."""
        references = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id == symbol:
                references.append({
                    'line': node.lineno,
                    'col': node.col_offset,
                    'context': node.ctx.__class__.__name__  # Load, Store, Del
                })
            elif isinstance(node, ast.Attribute) and node.attr == symbol:
                references.append({
                    'line': node.lineno,
                    'col': node.col_offset,
                    'context': 'Attribute'
                })

        return {'symbol': symbol, 'references': references, 'count': len(references)}

    def _extract_all_symbols(self, tree: ast.AST, source: str) -> Dict[str, Any]:
        """Extract all symbols in the file."""
        symbols = {
            'classes': [],
            'functions': [],
            'variables': [],
            'imports': []
        }

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                symbols['classes'].append({
                    'name': node.name,
                    'line': node.lineno,
                    'methods': [m.name for m in node.body if isinstance(m, ast.FunctionDef)]
                })
            elif isinstance(node, ast.FunctionDef):
                symbols['functions'].append({
                    'name': node.name,
                    'line': node.lineno,
                    'args': [arg.arg for arg in node.args.args]
                })
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    symbols['imports'].append({
                        'name': alias.name,
                        'alias': alias.asname,
                        'line': node.lineno
                    })

        return symbols

    def _generate_outline(self, tree: ast.AST, source: str) -> Dict[str, Any]:
        """Generate hierarchical outline of code structure."""
        outline = []

        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                class_info = {
                    'type': 'class',
                    'name': node.name,
                    'line': node.lineno,
                    'children': []
                }
                for child in node.body:
                    if isinstance(child, ast.FunctionDef):
                        class_info['children'].append({
                            'type': 'method',
                            'name': child.name,
                            'line': child.lineno
                        })
                outline.append(class_info)
            elif isinstance(node, ast.FunctionDef):
                outline.append({
                    'type': 'function',
                    'name': node.name,
                    'line': node.lineno
                })

        return {'outline': outline}


class RefactoringTool(Tool):
    """
    Advanced refactoring operations for code improvement.
    """

    name = "refactor_code"
    description = "Perform code refactoring operations (rename, extract, inline, etc.)"

    def execute(self, file_path: str, operation: str, **kwargs) -> Dict[str, Any]:
        """
        Execute refactoring operation.

        Args:
            file_path: Path to file
            operation: Refactoring operation (rename, extract_function, extract_variable, inline)
            **kwargs: Operation-specific parameters

        Returns:
            Dict with refactored code or error
        """
        try:
            if not os.path.exists(file_path):
                return {"error": f"File not found: {file_path}"}

            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()

            if operation == "rename":
                return self._rename_symbol(source_code, kwargs.get('old_name'),
                                          kwargs.get('new_name'))
            elif operation == "extract_function":
                return self._extract_function(source_code, kwargs.get('start_line'),
                                             kwargs.get('end_line'), kwargs.get('func_name'))
            elif operation == "extract_variable":
                return self._extract_variable(source_code, kwargs.get('expression'),
                                             kwargs.get('var_name'))
            elif operation == "remove_dead_code":
                return self._remove_dead_code(source_code)
            else:
                return {"error": f"Unknown operation: {operation}"}

        except Exception as e:
            return {"error": f"Refactoring failed: {str(e)}"}

    def _rename_symbol(self, source: str, old_name: str, new_name: str) -> Dict[str, Any]:
        """Rename a symbol throughout the code."""
        if not old_name or not new_name:
            return {"error": "Both old_name and new_name are required"}

        # Use word boundaries to avoid partial matches
        pattern = r'\b' + re.escape(old_name) + r'\b'
        new_source = re.sub(pattern, new_name, source)

        return {
            'success': True,
            'refactored_code': new_source,
            'changes': source.count(old_name)
        }

    def _extract_function(self, source: str, start_line: int, end_line: int,
                         func_name: str) -> Dict[str, Any]:
        """Extract code into a new function."""
        lines = source.split('\n')

        if start_line < 1 or end_line > len(lines) or start_line > end_line:
            return {"error": "Invalid line range"}

        extracted_lines = lines[start_line-1:end_line]
        extracted_code = '\n'.join(extracted_lines)

        # Create new function
        new_function = f"\ndef {func_name}():\n"
        for line in extracted_lines:
            new_function += f"    {line}\n"

        # Replace original code with function call
        lines[start_line-1:end_line] = [f"    {func_name}()"]

        # Insert function definition at the top
        lines.insert(0, new_function)

        return {
            'success': True,
            'refactored_code': '\n'.join(lines),
            'extracted_function': new_function
        }

    def _extract_variable(self, source: str, expression: str, var_name: str) -> Dict[str, Any]:
        """Extract expression into a variable."""
        if not expression or not var_name:
            return {"error": "Both expression and var_name are required"}

        # Find first occurrence and extract
        if expression in source:
            new_source = source.replace(expression, var_name, 1)
            new_source = f"{var_name} = {expression}\n" + new_source

            return {
                'success': True,
                'refactored_code': new_source
            }
        else:
            return {"error": f"Expression not found: {expression}"}

    def _remove_dead_code(self, source: str) -> Dict[str, Any]:
        """Remove unreachable code and unused imports."""
        try:
            tree = ast.parse(source)

            # Simple dead code detection
            dead_code_lines = set()

            for node in ast.walk(tree):
                # Detect code after return/break/continue
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    for i, stmt in enumerate(node.body):
                        if isinstance(stmt, (ast.Return, ast.Raise)):
                            # Mark all following statements as dead
                            for j in range(i + 1, len(node.body)):
                                if hasattr(node.body[j], 'lineno'):
                                    dead_code_lines.add(node.body[j].lineno)

            if dead_code_lines:
                lines = source.split('\n')
                cleaned_lines = [line for i, line in enumerate(lines, 1)
                               if i not in dead_code_lines]
                return {
                    'success': True,
                    'refactored_code': '\n'.join(cleaned_lines),
                    'removed_lines': list(dead_code_lines)
                }
            else:
                return {
                    'success': True,
                    'refactored_code': source,
                    'message': 'No dead code found'
                }

        except SyntaxError as e:
            return {"error": f"Syntax error: {str(e)}"}


class CodeFormatterTool(Tool):
    """
    Code formatting and style enforcement using Black and autopep8.
    """

    name = "format_code"
    description = "Format code using Black, autopep8, or isort"

    def execute(self, file_path: str = None, code: str = None,
                formatter: str = "black", line_length: int = 88) -> Dict[str, Any]:
        """
        Format code using specified formatter.

        Args:
            file_path: Path to file to format (optional if code provided)
            code: Code string to format (optional if file_path provided)
            formatter: Formatter to use (black, autopep8, isort)
            line_length: Maximum line length

        Returns:
            Dict with formatted code
        """
        try:
            if file_path:
                with open(file_path, 'r', encoding='utf-8') as f:
                    code = f.read()

            if not code:
                return {"error": "Either file_path or code must be provided"}

            if formatter == "black":
                return self._format_with_black(code, line_length)
            elif formatter == "autopep8":
                return self._format_with_autopep8(code)
            elif formatter == "isort":
                return self._format_with_isort(code)
            else:
                return {"error": f"Unknown formatter: {formatter}"}

        except Exception as e:
            return {"error": f"Formatting failed: {str(e)}"}

    def _format_with_black(self, code: str, line_length: int) -> Dict[str, Any]:
        """Format using Black."""
        try:
            import black

            mode = black.Mode(line_length=line_length)
            formatted = black.format_str(code, mode=mode)

            return {
                'success': True,
                'formatted_code': formatted,
                'formatter': 'black'
            }
        except ImportError:
            return {"error": "Black not installed. Install with: pip install black"}
        except Exception as e:
            return {"error": f"Black formatting failed: {str(e)}"}

    def _format_with_autopep8(self, code: str) -> Dict[str, Any]:
        """Format using autopep8."""
        try:
            import autopep8

            formatted = autopep8.fix_code(code)

            return {
                'success': True,
                'formatted_code': formatted,
                'formatter': 'autopep8'
            }
        except ImportError:
            return {"error": "autopep8 not installed. Install with: pip install autopep8"}
        except Exception as e:
            return {"error": f"autopep8 formatting failed: {str(e)}"}

    def _format_with_isort(self, code: str) -> Dict[str, Any]:
        """Sort imports using isort."""
        try:
            import isort

            formatted = isort.code(code)

            return {
                'success': True,
                'formatted_code': formatted,
                'formatter': 'isort'
            }
        except ImportError:
            return {"error": "isort not installed. Install with: pip install isort"}
        except Exception as e:
            return {"error": f"isort failed: {str(e)}"}


class LiveLintTool(Tool):
    """
    Real-time linting with multiple linters (pylint, flake8, mypy).
    """

    name = "live_lint"
    description = "Run linting analysis with pylint, flake8, or mypy"

    def execute(self, file_path: str, linter: str = "pylint",
                config: Optional[str] = None) -> Dict[str, Any]:
        """
        Run linting analysis.

        Args:
            file_path: Path to file
            linter: Linter to use (pylint, flake8, mypy)
            config: Optional config file path

        Returns:
            Dict with linting results
        """
        try:
            if not os.path.exists(file_path):
                return {"error": f"File not found: {file_path}"}

            if linter == "pylint":
                return self._run_pylint(file_path, config)
            elif linter == "flake8":
                return self._run_flake8(file_path, config)
            elif linter == "mypy":
                return self._run_mypy(file_path, config)
            else:
                return {"error": f"Unknown linter: {linter}"}

        except Exception as e:
            return {"error": f"Linting failed: {str(e)}"}

    def _run_pylint(self, file_path: str, config: Optional[str]) -> Dict[str, Any]:
        """Run pylint analysis."""
        cmd = ["pylint", "--output-format=json"]
        if config:
            cmd.extend(["--rcfile", config])
        cmd.append(file_path)

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            # Parse JSON output
            try:
                issues = json.loads(result.stdout) if result.stdout else []
            except json.JSONDecodeError:
                issues = []

            return {
                'success': True,
                'linter': 'pylint',
                'issues': issues,
                'count': len(issues),
                'file': file_path
            }
        except FileNotFoundError:
            return {"error": "pylint not found. Install with: pip install pylint"}
        except subprocess.TimeoutExpired:
            return {"error": "Pylint timed out"}

    def _run_flake8(self, file_path: str, config: Optional[str]) -> Dict[str, Any]:
        """Run flake8 analysis."""
        cmd = ["flake8", "--format=json"]
        if config:
            cmd.extend(["--config", config])
        cmd.append(file_path)

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            # Parse output
            issues = []
            for line in result.stdout.split('\n'):
                if line.strip():
                    issues.append({'message': line})

            return {
                'success': True,
                'linter': 'flake8',
                'issues': issues,
                'count': len(issues),
                'file': file_path
            }
        except FileNotFoundError:
            return {"error": "flake8 not found. Install with: pip install flake8"}
        except subprocess.TimeoutExpired:
            return {"error": "flake8 timed out"}

    def _run_mypy(self, file_path: str, config: Optional[str]) -> Dict[str, Any]:
        """Run mypy type checking."""
        cmd = ["mypy", "--show-error-codes"]
        if config:
            cmd.extend(["--config-file", config])
        cmd.append(file_path)

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            # Parse output
            issues = []
            for line in result.stdout.split('\n'):
                if line.strip() and ':' in line:
                    issues.append({'message': line})

            return {
                'success': True,
                'linter': 'mypy',
                'issues': issues,
                'count': len(issues),
                'file': file_path
            }
        except FileNotFoundError:
            return {"error": "mypy not found. Install with: pip install mypy"}
        except subprocess.TimeoutExpired:
            return {"error": "mypy timed out"}


class SymbolNavigationTool(Tool):
    """
    Navigate code symbols across entire project (go-to-definition, find-references).
    """

    name = "navigate_symbols"
    description = "Navigate symbols across project (definitions, references, implementations)"

    def execute(self, project_root: str, symbol: str,
                action: str = "find_all") -> Dict[str, Any]:
        """
        Navigate symbols across project.

        Args:
            project_root: Root directory of project
            symbol: Symbol to find
            action: Action (find_all, definitions, references)

        Returns:
            Dict with symbol locations
        """
        try:
            if not os.path.isdir(project_root):
                return {"error": f"Directory not found: {project_root}"}

            results = {
                'symbol': symbol,
                'definitions': [],
                'references': [],
                'files_searched': 0
            }

            # Search all Python files
            for root, dirs, files in os.walk(project_root):
                # Skip common non-code directories
                dirs[:] = [d for d in dirs if d not in {'.git', '__pycache__', 'venv', 'env', 'node_modules'}]

                for file in files:
                    if file.endswith('.py'):
                        file_path = os.path.join(root, file)
                        results['files_searched'] += 1

                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                source = f.read()

                            tree = ast.parse(source, filename=file_path)

                            # Find definitions
                            for node in ast.walk(tree):
                                if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == symbol:
                                    results['definitions'].append({
                                        'file': file_path,
                                        'line': node.lineno,
                                        'type': 'function' if isinstance(node, ast.FunctionDef) else 'class'
                                    })
                                elif isinstance(node, ast.Name) and node.id == symbol:
                                    results['references'].append({
                                        'file': file_path,
                                        'line': node.lineno,
                                        'context': node.ctx.__class__.__name__
                                    })
                        except (SyntaxError, UnicodeDecodeError):
                            # Skip files with errors
                            continue

            results['total_definitions'] = len(results['definitions'])
            results['total_references'] = len(results['references'])

            return results

        except Exception as e:
            return {"error": f"Symbol navigation failed: {str(e)}"}


class CodeCompletionTool(Tool):
    """
    AI-powered code completion suggestions.
    """

    name = "code_completion"
    description = "Generate code completion suggestions using context"

    def execute(self, file_path: str, line: int, column: int,
                context_lines: int = 10) -> Dict[str, Any]:
        """
        Generate code completions.

        Args:
            file_path: Path to file
            line: Current line number
            column: Current column number
            context_lines: Number of context lines to consider

        Returns:
            Dict with completion suggestions
        """
        try:
            if not os.path.exists(file_path):
                return {"error": f"File not found: {file_path}"}

            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            if line < 1 or line > len(lines):
                return {"error": f"Invalid line number: {line}"}

            # Get context
            start = max(0, line - context_lines - 1)
            end = min(len(lines), line + context_lines)
            context = ''.join(lines[start:end])

            current_line = lines[line - 1]
            prefix = current_line[:column]

            # Simple completion suggestions based on context
            suggestions = self._generate_suggestions(context, prefix, file_path)

            return {
                'success': True,
                'suggestions': suggestions,
                'line': line,
                'column': column,
                'prefix': prefix.strip()
            }

        except Exception as e:
            return {"error": f"Code completion failed: {str(e)}"}

    def _generate_suggestions(self, context: str, prefix: str,
                            file_path: str) -> List[Dict[str, str]]:
        """Generate completion suggestions."""
        suggestions = []

        try:
            tree = ast.parse(context)

            # Extract available symbols
            symbols = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    symbols.add(node.name)
                elif isinstance(node, ast.ClassDef):
                    symbols.add(node.name)
                elif isinstance(node, ast.Name):
                    symbols.add(node.id)
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    for alias in node.names:
                        symbols.add(alias.name)

            # Filter symbols that match prefix
            prefix_stripped = prefix.strip()
            for symbol in symbols:
                if symbol.startswith(prefix_stripped):
                    suggestions.append({
                        'label': symbol,
                        'kind': 'variable',
                        'detail': f'From {os.path.basename(file_path)}'
                    })

            # Add common Python keywords if they match
            keywords = ['def', 'class', 'if', 'else', 'elif', 'for', 'while',
                       'try', 'except', 'finally', 'with', 'import', 'from',
                       'return', 'yield', 'async', 'await']

            for keyword in keywords:
                if keyword.startswith(prefix_stripped):
                    suggestions.append({
                        'label': keyword,
                        'kind': 'keyword',
                        'detail': 'Python keyword'
                    })

        except SyntaxError:
            # If there's a syntax error, just provide keyword suggestions
            pass

        return suggestions[:20]  # Limit to top 20 suggestions
