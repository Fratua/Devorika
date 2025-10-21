"""
Advanced Codebase Analysis Tools for Devorika
Dependency graphs, impact analysis, architectural insights, and code quality metrics.
"""

import os
import ast
import json
import re
from typing import Dict, Any, List, Set, Tuple, Optional
from pathlib import Path
from collections import defaultdict
from .base import Tool


class DependencyGraphTool(Tool):
    """
    Generate dependency graphs for code modules.
    """

    name = "dependency_graph"
    description = "Generate and analyze code dependency graphs"

    def execute(self, project_dir: str = ".", output_format: str = "json",
                output_file: str = "dependencies.json") -> Dict[str, Any]:
        """
        Generate dependency graph.

        Args:
            project_dir: Project directory
            output_format: Output format (json, dot, mermaid)
            output_file: Output file path

        Returns:
            Dict with dependency graph
        """
        try:
            dependencies = defaultdict(set)
            all_files = []

            # Scan all Python files
            for root, dirs, files in os.walk(project_dir):
                # Skip common directories
                dirs[:] = [d for d in dirs if d not in {'.git', '__pycache__', 'venv', 'env', 'node_modules'}]

                for file in files:
                    if file.endswith('.py'):
                        file_path = os.path.join(root, file)
                        all_files.append(file_path)

                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                tree = ast.parse(f.read(), filename=file_path)

                            # Extract imports
                            for node in ast.walk(tree):
                                if isinstance(node, ast.Import):
                                    for alias in node.names:
                                        dependencies[file_path].add(alias.name)
                                elif isinstance(node, ast.ImportFrom):
                                    if node.module:
                                        dependencies[file_path].add(node.module)

                        except (SyntaxError, UnicodeDecodeError):
                            continue

            # Convert to serializable format
            dep_dict = {k: list(v) for k, v in dependencies.items()}

            # Calculate metrics
            total_files = len(all_files)
            total_dependencies = sum(len(deps) for deps in dependencies.values())
            avg_dependencies = total_dependencies / total_files if total_files > 0 else 0

            # Find circular dependencies
            circular = self._find_circular_dependencies(dependencies)

            # Generate output based on format
            if output_format == "json":
                output_data = {
                    'dependencies': dep_dict,
                    'metrics': {
                        'total_files': total_files,
                        'total_dependencies': total_dependencies,
                        'avg_dependencies_per_file': avg_dependencies,
                        'circular_dependencies': len(circular)
                    },
                    'circular_dependencies': circular
                }

                with open(output_file, 'w') as f:
                    json.dump(output_data, f, indent=2)

            elif output_format == "dot":
                # Generate Graphviz DOT format
                dot_content = self._generate_dot_graph(dependencies)
                with open(output_file, 'w') as f:
                    f.write(dot_content)

            elif output_format == "mermaid":
                # Generate Mermaid diagram
                mermaid_content = self._generate_mermaid_graph(dependencies)
                with open(output_file, 'w') as f:
                    f.write(mermaid_content)

            return {
                'success': True,
                'project_dir': project_dir,
                'total_files': total_files,
                'total_dependencies': total_dependencies,
                'avg_dependencies': avg_dependencies,
                'circular_dependencies_count': len(circular),
                'output_file': output_file,
                'output_format': output_format
            }

        except Exception as e:
            return {"error": f"Dependency graph generation failed: {str(e)}"}

    def _find_circular_dependencies(self, deps: Dict[str, Set[str]]) -> List[List[str]]:
        """Find circular dependencies using DFS."""
        circular = []
        visited = set()
        rec_stack = set()

        def dfs(node, path):
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in deps.get(node, []):
                if neighbor not in visited:
                    if dfs(neighbor, path):
                        return True
                elif neighbor in rec_stack:
                    # Found a cycle
                    cycle_start = path.index(neighbor)
                    circular.append(path[cycle_start:])
                    return True

            path.pop()
            rec_stack.remove(node)
            return False

        for node in deps:
            if node not in visited:
                dfs(node, [])

        return circular

    def _generate_dot_graph(self, deps: Dict[str, Set[str]]) -> str:
        """Generate Graphviz DOT format."""
        dot = "digraph Dependencies {\n"
        dot += "  rankdir=LR;\n"
        dot += "  node [shape=box];\n\n"

        for file, imports in deps.items():
            file_name = os.path.basename(file)
            for imp in imports:
                dot += f'  "{file_name}" -> "{imp}";\n'

        dot += "}\n"
        return dot

    def _generate_mermaid_graph(self, deps: Dict[str, Set[str]]) -> str:
        """Generate Mermaid diagram."""
        mermaid = "graph LR\n"

        for file, imports in deps.items():
            file_name = os.path.basename(file).replace('.py', '')
            for imp in imports:
                imp_name = imp.replace('.', '_')
                mermaid += f"  {file_name} --> {imp_name}\n"

        return mermaid


class ImpactAnalysisTool(Tool):
    """
    Analyze the impact of code changes across the codebase.
    """

    name = "impact_analysis"
    description = "Analyze impact of code changes across codebase"

    def execute(self, file_path: str, symbol: str, project_dir: str = ".") -> Dict[str, Any]:
        """
        Analyze impact of changing a symbol.

        Args:
            file_path: File containing the symbol
            symbol: Symbol name (function, class, variable)
            project_dir: Project directory

        Returns:
            Dict with impact analysis
        """
        try:
            impacted_files = []
            total_references = 0

            # Search for references across project
            for root, dirs, files in os.walk(project_dir):
                dirs[:] = [d for d in dirs if d not in {'.git', '__pycache__', 'venv'}]

                for file in files:
                    if file.endswith('.py'):
                        current_file = os.path.join(root, file)

                        try:
                            with open(current_file, 'r', encoding='utf-8') as f:
                                content = f.read()

                            # Count references
                            pattern = r'\b' + re.escape(symbol) + r'\b'
                            matches = re.findall(pattern, content)

                            if matches and current_file != file_path:
                                impacted_files.append({
                                    'file': current_file,
                                    'references': len(matches),
                                    'lines': [i + 1 for i, line in enumerate(content.split('\n'))
                                            if re.search(pattern, line)]
                                })
                                total_references += len(matches)

                        except (UnicodeDecodeError, PermissionError):
                            continue

            # Calculate risk score
            risk_score = min(100, (total_references * 10) + (len(impacted_files) * 5))

            return {
                'success': True,
                'symbol': symbol,
                'source_file': file_path,
                'impacted_files': impacted_files,
                'total_impacted_files': len(impacted_files),
                'total_references': total_references,
                'risk_score': risk_score,
                'risk_level': 'HIGH' if risk_score > 70 else 'MEDIUM' if risk_score > 30 else 'LOW'
            }

        except Exception as e:
            return {"error": f"Impact analysis failed: {str(e)}"}


class ArchitectureAnalyzerTool(Tool):
    """
    Analyze software architecture and design patterns.
    """

    name = "architecture_analyzer"
    description = "Analyze software architecture and identify design patterns"

    def execute(self, project_dir: str = ".") -> Dict[str, Any]:
        """
        Analyze project architecture.

        Args:
            project_dir: Project directory

        Returns:
            Dict with architecture analysis
        """
        try:
            architecture = {
                'layers': defaultdict(list),
                'patterns': [],
                'modules': {},
                'complexity': {}
            }

            # Scan project structure
            for root, dirs, files in os.walk(project_dir):
                dirs[:] = [d for d in dirs if d not in {'.git', '__pycache__', 'venv'}]

                for file in files:
                    if file.endswith('.py'):
                        file_path = os.path.join(root, file)
                        rel_path = os.path.relpath(file_path, project_dir)

                        # Identify layer based on directory structure
                        layer = self._identify_layer(rel_path)
                        architecture['layers'][layer].append(rel_path)

                        # Analyze module
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                                tree = ast.parse(content)

                            module_info = {
                                'classes': len([n for n in tree.body if isinstance(n, ast.ClassDef)]),
                                'functions': len([n for n in tree.body if isinstance(n, ast.FunctionDef)]),
                                'loc': len(content.split('\n'))
                            }
                            architecture['modules'][rel_path] = module_info

                            # Detect design patterns
                            patterns = self._detect_patterns(tree, content)
                            if patterns:
                                architecture['patterns'].extend([{
                                    'file': rel_path,
                                    'pattern': p
                                } for p in patterns])

                        except (SyntaxError, UnicodeDecodeError):
                            continue

            # Calculate architecture metrics
            total_files = sum(len(files) for files in architecture['layers'].values())
            avg_loc = sum(m['loc'] for m in architecture['modules'].values()) / total_files if total_files > 0 else 0

            return {
                'success': True,
                'project_dir': project_dir,
                'layers': dict(architecture['layers']),
                'layer_distribution': {k: len(v) for k, v in architecture['layers'].items()},
                'design_patterns': architecture['patterns'],
                'total_modules': len(architecture['modules']),
                'average_loc': avg_loc,
                'architecture_score': self._calculate_architecture_score(architecture)
            }

        except Exception as e:
            return {"error": f"Architecture analysis failed: {str(e)}"}

    def _identify_layer(self, file_path: str) -> str:
        """Identify architectural layer based on path."""
        path_lower = file_path.lower()

        if any(x in path_lower for x in ['model', 'models', 'entity', 'entities']):
            return 'data'
        elif any(x in path_lower for x in ['view', 'views', 'template', 'ui']):
            return 'presentation'
        elif any(x in path_lower for x in ['controller', 'route', 'api', 'handler']):
            return 'controller'
        elif any(x in path_lower for x in ['service', 'business', 'logic']):
            return 'business'
        elif any(x in path_lower for x in ['repository', 'dao', 'database', 'db']):
            return 'data_access'
        elif any(x in path_lower for x in ['util', 'helper', 'common']):
            return 'utility'
        else:
            return 'other'

    def _detect_patterns(self, tree: ast.AST, content: str) -> List[str]:
        """Detect common design patterns."""
        patterns = []

        # Singleton pattern
        if re.search(r'class\s+\w+.*:\s+_instance\s*=\s*None', content):
            patterns.append('Singleton')

        # Factory pattern
        if re.search(r'def\s+create_\w+|def\s+\w+_factory', content):
            patterns.append('Factory')

        # Observer pattern
        if re.search(r'def\s+notify|def\s+subscribe|def\s+observe', content):
            patterns.append('Observer')

        # Decorator pattern
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and len(node.decorator_list) > 0:
                patterns.append('Decorator')
                break

        # Strategy pattern
        if re.search(r'class\s+\w+Strategy', content):
            patterns.append('Strategy')

        return list(set(patterns))  # Remove duplicates

    def _calculate_architecture_score(self, arch: Dict[str, Any]) -> float:
        """Calculate architecture quality score."""
        score = 100.0

        # Penalize for unbalanced layers
        total_files = sum(len(files) for files in arch['layers'].values())
        if total_files > 0:
            for layer, files in arch['layers'].items():
                ratio = len(files) / total_files
                if ratio > 0.5:  # One layer has more than 50% of files
                    score -= 10

        # Reward for design patterns
        unique_patterns = len(set(p['pattern'] for p in arch['patterns']))
        score += min(20, unique_patterns * 5)

        # Penalize for high complexity
        avg_classes = sum(m['classes'] for m in arch['modules'].values()) / len(arch['modules']) if arch['modules'] else 0
        if avg_classes > 5:
            score -= 10

        return max(0, min(100, score))


class CodeQualityMetricsTool(Tool):
    """
    Comprehensive code quality metrics and technical debt analysis.
    """

    name = "code_quality_metrics"
    description = "Calculate comprehensive code quality metrics"

    def execute(self, project_dir: str = ".", output_file: str = "quality_report.json") -> Dict[str, Any]:
        """
        Calculate code quality metrics.

        Args:
            project_dir: Project directory
            output_file: Output report file

        Returns:
            Dict with quality metrics
        """
        try:
            metrics = {
                'maintainability_index': [],
                'cyclomatic_complexity': [],
                'code_duplication': [],
                'comment_ratio': [],
                'test_coverage_estimate': 0,
                'technical_debt_minutes': 0
            }

            total_files = 0
            total_loc = 0
            total_comments = 0

            for root, dirs, files in os.walk(project_dir):
                dirs[:] = [d for d in dirs if d not in {'.git', '__pycache__', 'venv'}]

                for file in files:
                    if file.endswith('.py'):
                        file_path = os.path.join(root, file)
                        total_files += 1

                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                                lines = content.split('\n')

                            total_loc += len(lines)

                            # Count comments
                            comments = sum(1 for line in lines if line.strip().startswith('#'))
                            total_comments += comments

                            # Parse AST
                            tree = ast.parse(content)

                            # Cyclomatic complexity
                            complexity = self._calculate_complexity(tree)
                            metrics['cyclomatic_complexity'].append({
                                'file': file_path,
                                'complexity': complexity
                            })

                            # Maintainability index (simplified)
                            mi = self._calculate_maintainability_index(len(lines), complexity, comments)
                            metrics['maintainability_index'].append({
                                'file': file_path,
                                'index': mi
                            })

                        except (SyntaxError, UnicodeDecodeError):
                            continue

            # Calculate aggregated metrics
            avg_complexity = sum(m['complexity'] for m in metrics['cyclomatic_complexity']) / total_files if total_files > 0 else 0
            avg_mi = sum(m['index'] for m in metrics['maintainability_index']) / total_files if total_files > 0 else 0
            comment_ratio = (total_comments / total_loc * 100) if total_loc > 0 else 0

            # Estimate technical debt (simplified)
            technical_debt = self._estimate_technical_debt(avg_complexity, avg_mi, comment_ratio)

            report = {
                'project_dir': project_dir,
                'total_files': total_files,
                'total_loc': total_loc,
                'avg_cyclomatic_complexity': avg_complexity,
                'avg_maintainability_index': avg_mi,
                'comment_ratio_percent': comment_ratio,
                'technical_debt_hours': technical_debt / 60,
                'quality_grade': self._get_quality_grade(avg_mi),
                'recommendations': self._get_recommendations(avg_complexity, avg_mi, comment_ratio)
            }

            # Save report
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)

            return {
                'success': True,
                **report,
                'report_file': output_file
            }

        except Exception as e:
            return {"error": f"Code quality analysis failed: {str(e)}"}

    def _calculate_complexity(self, tree: ast.AST) -> int:
        """Calculate cyclomatic complexity."""
        complexity = 1

        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1

        return complexity

    def _calculate_maintainability_index(self, loc: int, complexity: int, comments: int) -> float:
        """Calculate maintainability index (simplified)."""
        import math

        # Halstead Volume (simplified estimate)
        volume = loc * 4.2

        # MI = 171 - 5.2 * ln(V) - 0.23 * CC - 16.2 * ln(LOC) + 50 * sin(sqrt(2.4 * CM))
        if volume > 0 and loc > 0:
            mi = 171 - 5.2 * math.log(volume) - 0.23 * complexity - 16.2 * math.log(loc)
            mi = max(0, min(100, mi))
        else:
            mi = 50

        return mi

    def _estimate_technical_debt(self, complexity: float, mi: float, comment_ratio: float) -> float:
        """Estimate technical debt in minutes."""
        debt = 0

        # High complexity penalty
        if complexity > 10:
            debt += (complexity - 10) * 15

        # Low maintainability penalty
        if mi < 65:
            debt += (65 - mi) * 2

        # Low documentation penalty
        if comment_ratio < 10:
            debt += (10 - comment_ratio) * 5

        return debt

    def _get_quality_grade(self, mi: float) -> str:
        """Get quality grade based on maintainability index."""
        if mi >= 85:
            return 'A'
        elif mi >= 75:
            return 'B'
        elif mi >= 65:
            return 'C'
        elif mi >= 55:
            return 'D'
        else:
            return 'F'

    def _get_recommendations(self, complexity: float, mi: float, comment_ratio: float) -> List[str]:
        """Get improvement recommendations."""
        recommendations = []

        if complexity > 10:
            recommendations.append("Reduce cyclomatic complexity by breaking down complex functions")

        if mi < 65:
            recommendations.append("Improve maintainability by refactoring and adding documentation")

        if comment_ratio < 10:
            recommendations.append("Add more comments and documentation to improve code clarity")

        if not recommendations:
            recommendations.append("Code quality is good! Keep up the excellent work")

        return recommendations


class CodeDuplicationDetectorTool(Tool):
    """
    Detect code duplication and suggest refactoring.
    """

    name = "code_duplication_detector"
    description = "Detect duplicate code blocks across codebase"

    def execute(self, project_dir: str = ".", min_lines: int = 5) -> Dict[str, Any]:
        """
        Detect code duplication.

        Args:
            project_dir: Project directory
            min_lines: Minimum lines for duplication detection

        Returns:
            Dict with duplication analysis
        """
        try:
            code_blocks = defaultdict(list)
            duplicates = []

            # Scan all Python files
            for root, dirs, files in os.walk(project_dir):
                dirs[:] = [d for d in dirs if d not in {'.git', '__pycache__', 'venv'}]

                for file in files:
                    if file.endswith('.py'):
                        file_path = os.path.join(root, file)

                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                lines = f.readlines()

                            # Extract code blocks
                            for i in range(len(lines) - min_lines + 1):
                                block = ''.join(lines[i:i + min_lines])
                                # Normalize whitespace
                                normalized = ' '.join(block.split())

                                if len(normalized) > 20:  # Skip very short blocks
                                    code_blocks[normalized].append({
                                        'file': file_path,
                                        'start_line': i + 1,
                                        'end_line': i + min_lines
                                    })

                        except (UnicodeDecodeError, PermissionError):
                            continue

            # Find duplicates
            for block_hash, occurrences in code_blocks.items():
                if len(occurrences) > 1:
                    duplicates.append({
                        'occurrences': occurrences,
                        'count': len(occurrences),
                        'preview': block_hash[:100] + '...' if len(block_hash) > 100 else block_hash
                    })

            # Sort by number of occurrences
            duplicates.sort(key=lambda x: x['count'], reverse=True)

            return {
                'success': True,
                'project_dir': project_dir,
                'total_duplicates': len(duplicates),
                'top_duplicates': duplicates[:10],  # Top 10 most duplicated
                'duplication_score': min(100, len(duplicates) * 2),
                'recommendation': 'Extract common code into reusable functions' if duplicates else 'No significant duplication found'
            }

        except Exception as e:
            return {"error": f"Duplication detection failed: {str(e)}"}
