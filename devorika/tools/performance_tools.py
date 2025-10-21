"""
Performance Profiling and Optimization Tools for Devorika
CPU, memory, I/O profiling, and performance analysis.
"""

import os
import sys
import time
import json
import subprocess
from typing import Dict, Any, Optional, Callable
from .base import Tool


class CPUProfilerTool(Tool):
    """
    CPU profiling using cProfile and py-spy.
    """

    name = "cpu_profiler"
    description = "Profile CPU usage and execution time"

    def execute(self, target: str, profiler: str = "cprofile",
                duration: int = 10, **kwargs) -> Dict[str, Any]:
        """
        Profile CPU usage.

        Args:
            target: Python file or function to profile
            profiler: Profiler to use (cprofile, pyspy)
            duration: Duration in seconds for sampling profilers
            **kwargs: Additional parameters

        Returns:
            Dict with profiling results
        """
        try:
            if profiler == "cprofile":
                return self._cprofile(target, kwargs)
            elif profiler == "pyspy":
                return self._pyspy(target, duration, kwargs)
            else:
                return {"error": f"Unknown profiler: {profiler}"}

        except Exception as e:
            return {"error": f"CPU profiling failed: {str(e)}"}

    def _cprofile(self, target: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Profile using cProfile."""
        import cProfile
        import pstats
        from io import StringIO

        if not os.path.exists(target):
            return {"error": f"Target file not found: {target}"}

        # Create profiler
        profiler = cProfile.Profile()

        # Profile the script
        try:
            profiler.run(f'exec(open("{target}").read())')

            # Get stats
            stats_stream = StringIO()
            stats = pstats.Stats(profiler, stream=stats_stream)
            stats.strip_dirs()
            stats.sort_stats('cumulative')
            stats.print_stats(20)  # Top 20 functions

            return {
                'success': True,
                'profiler': 'cProfile',
                'target': target,
                'stats': stats_stream.getvalue(),
                'total_calls': stats.total_calls,
                'primitive_calls': stats.prim_calls
            }

        except Exception as e:
            return {"error": f"cProfile error: {str(e)}"}

    def _pyspy(self, target: str, duration: int, params: Dict[str, Any]) -> Dict[str, Any]:
        """Profile using py-spy."""
        output_file = params.get('output_file', 'profile.svg')

        cmd = [
            "py-spy", "record",
            "-o", output_file,
            "-d", str(duration),
            "--", "python", target
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=duration + 10)

            if result.returncode == 0:
                return {
                    'success': True,
                    'profiler': 'py-spy',
                    'target': target,
                    'duration': duration,
                    'output_file': output_file,
                    'message': f'Flame graph saved to {output_file}'
                }
            else:
                return {"error": f"py-spy failed: {result.stderr}"}

        except FileNotFoundError:
            return {"error": "py-spy not installed. Install with: pip install py-spy"}
        except subprocess.TimeoutExpired:
            return {"error": "py-spy timed out"}


class MemoryProfilerTool(Tool):
    """
    Memory profiling and leak detection.
    """

    name = "memory_profiler"
    description = "Profile memory usage and detect memory leaks"

    def execute(self, target: str, profiler: str = "memory_profiler",
                **kwargs) -> Dict[str, Any]:
        """
        Profile memory usage.

        Args:
            target: Python file to profile
            profiler: Profiler to use (memory_profiler, tracemalloc)
            **kwargs: Additional parameters

        Returns:
            Dict with memory profiling results
        """
        try:
            if profiler == "memory_profiler":
                return self._memory_profiler(target)
            elif profiler == "tracemalloc":
                return self._tracemalloc(target)
            else:
                return {"error": f"Unknown profiler: {profiler}"}

        except Exception as e:
            return {"error": f"Memory profiling failed: {str(e)}"}

    def _memory_profiler(self, target: str) -> Dict[str, Any]:
        """Profile using memory_profiler."""
        cmd = ["python", "-m", "memory_profiler", target]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

            return {
                'success': True,
                'profiler': 'memory_profiler',
                'target': target,
                'output': result.stdout
            }

        except FileNotFoundError:
            return {"error": "memory_profiler not installed. Install with: pip install memory-profiler"}
        except subprocess.TimeoutExpired:
            return {"error": "Memory profiling timed out"}

    def _tracemalloc(self, target: str) -> Dict[str, Any]:
        """Profile using tracemalloc."""
        import tracemalloc

        if not os.path.exists(target):
            return {"error": f"Target file not found: {target}"}

        tracemalloc.start()

        try:
            # Execute the script
            with open(target, 'r', encoding='utf-8') as f:
                code = f.read()
                exec(code)

            # Get snapshot
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('lineno')

            # Format results
            stats_list = []
            for stat in top_stats[:10]:
                stats_list.append({
                    'file': stat.traceback.format()[0],
                    'size': stat.size,
                    'size_mb': stat.size / 1024 / 1024,
                    'count': stat.count
                })

            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            return {
                'success': True,
                'profiler': 'tracemalloc',
                'target': target,
                'current_memory_mb': current / 1024 / 1024,
                'peak_memory_mb': peak / 1024 / 1024,
                'top_allocations': stats_list
            }

        except Exception as e:
            tracemalloc.stop()
            return {"error": f"tracemalloc error: {str(e)}"}


class PerformanceBenchmarkTool(Tool):
    """
    Benchmark code performance and compare implementations.
    """

    name = "performance_benchmark"
    description = "Benchmark and compare code performance"

    def execute(self, code: str, iterations: int = 1000,
                name: str = "benchmark") -> Dict[str, Any]:
        """
        Benchmark code performance.

        Args:
            code: Code to benchmark
            iterations: Number of iterations
            name: Benchmark name

        Returns:
            Dict with benchmark results
        """
        try:
            import timeit

            # Run benchmark
            total_time = timeit.timeit(code, number=iterations)
            avg_time = total_time / iterations

            return {
                'success': True,
                'name': name,
                'iterations': iterations,
                'total_time_seconds': total_time,
                'average_time_seconds': avg_time,
                'average_time_ms': avg_time * 1000,
                'operations_per_second': iterations / total_time
            }

        except Exception as e:
            return {"error": f"Benchmark failed: {str(e)}"}


class LoadTesterTool(Tool):
    """
    Load testing for web applications and APIs.
    """

    name = "load_tester"
    description = "Perform load testing on web applications"

    def execute(self, url: str, requests: int = 100, concurrent: int = 10,
                method: str = "GET", **kwargs) -> Dict[str, Any]:
        """
        Perform load testing.

        Args:
            url: Target URL
            requests: Total number of requests
            concurrent: Concurrent requests
            method: HTTP method
            **kwargs: Additional parameters (headers, data, etc.)

        Returns:
            Dict with load test results
        """
        try:
            import requests as req
            from concurrent.futures import ThreadPoolExecutor
            import statistics

            results = {
                'response_times': [],
                'status_codes': {},
                'errors': 0
            }

            def make_request():
                start = time.time()
                try:
                    if method == "GET":
                        response = req.get(url, timeout=10)
                    elif method == "POST":
                        response = req.post(url, json=kwargs.get('data'), timeout=10)
                    else:
                        return None

                    elapsed = time.time() - start
                    status = response.status_code

                    return {'time': elapsed, 'status': status, 'error': None}
                except Exception as e:
                    elapsed = time.time() - start
                    return {'time': elapsed, 'status': None, 'error': str(e)}

            # Execute load test
            with ThreadPoolExecutor(max_workers=concurrent) as executor:
                futures = [executor.submit(make_request) for _ in range(requests)]

                for future in futures:
                    result = future.result()
                    if result:
                        results['response_times'].append(result['time'])

                        if result['error']:
                            results['errors'] += 1
                        else:
                            status = result['status']
                            results['status_codes'][status] = results['status_codes'].get(status, 0) + 1

            # Calculate statistics
            response_times = results['response_times']

            return {
                'success': True,
                'url': url,
                'total_requests': requests,
                'concurrent_requests': concurrent,
                'completed_requests': len(response_times),
                'failed_requests': results['errors'],
                'status_codes': results['status_codes'],
                'avg_response_time': statistics.mean(response_times),
                'min_response_time': min(response_times),
                'max_response_time': max(response_times),
                'median_response_time': statistics.median(response_times),
                'requests_per_second': requests / sum(response_times) if sum(response_times) > 0 else 0
            }

        except ImportError:
            return {"error": "requests library not installed"}
        except Exception as e:
            return {"error": f"Load testing failed: {str(e)}"}


class CodeOptimizationTool(Tool):
    """
    Suggest code optimizations based on profiling and static analysis.
    """

    name = "code_optimization"
    description = "Analyze code and suggest performance optimizations"

    def execute(self, file_path: str) -> Dict[str, Any]:
        """
        Analyze code for optimization opportunities.

        Args:
            file_path: Path to Python file

        Returns:
            Dict with optimization suggestions
        """
        try:
            if not os.path.exists(file_path):
                return {"error": f"File not found: {file_path}"}

            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()

            suggestions = []

            # Check for common performance issues
            suggestions.extend(self._check_loops(source_code))
            suggestions.extend(self._check_data_structures(source_code))
            suggestions.extend(self._check_string_operations(source_code))
            suggestions.extend(self._check_function_calls(source_code))

            return {
                'success': True,
                'file': file_path,
                'suggestions': suggestions,
                'count': len(suggestions)
            }

        except Exception as e:
            return {"error": f"Code optimization analysis failed: {str(e)}"}

    def _check_loops(self, source: str) -> list:
        """Check for inefficient loops."""
        import re
        suggestions = []

        # Check for list concatenation in loops
        if re.search(r'for\s+.*:\s+.*\+=\s*\[', source):
            suggestions.append({
                'type': 'Loop Optimization',
                'issue': 'List concatenation in loop',
                'suggestion': 'Use list.append() or list comprehension instead',
                'impact': 'HIGH'
            })

        # Check for repeated function calls in loop condition
        if re.search(r'for\s+\w+\s+in\s+range\(len\(.*\)\)', source):
            suggestions.append({
                'type': 'Loop Optimization',
                'issue': 'len() called in loop',
                'suggestion': 'Store len() result in variable before loop',
                'impact': 'MEDIUM'
            })

        return suggestions

    def _check_data_structures(self, source: str) -> list:
        """Check for inefficient data structure usage."""
        import re
        suggestions = []

        # Check for membership testing on lists
        if re.search(r'if\s+\w+\s+in\s+\[.*\]', source):
            suggestions.append({
                'type': 'Data Structure',
                'issue': 'Membership test on list literal',
                'suggestion': 'Use set or tuple for better performance',
                'impact': 'MEDIUM'
            })

        return suggestions

    def _check_string_operations(self, source: str) -> list:
        """Check for inefficient string operations."""
        import re
        suggestions = []

        # Check for string concatenation in loops
        if re.search(r'for\s+.*:\s+.*\+\s*["\']', source):
            suggestions.append({
                'type': 'String Operations',
                'issue': 'String concatenation in loop',
                'suggestion': 'Use str.join() or list of strings',
                'impact': 'HIGH'
            })

        return suggestions

    def _check_function_calls(self, source: str) -> list:
        """Check for expensive function calls."""
        import re
        suggestions = []

        # Check for repeated compile calls
        if source.count('re.compile') > 3:
            suggestions.append({
                'type': 'Function Calls',
                'issue': 'Multiple regex compilations',
                'suggestion': 'Compile regex patterns once at module level',
                'impact': 'MEDIUM'
            })

        return suggestions
