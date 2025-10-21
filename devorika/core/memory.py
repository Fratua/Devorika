"""
Long-term Memory System
Allows Devorika to learn from past tasks and improve over time
"""

import json
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path


class Memory:
    """Long-term memory system for learning and context retention."""

    def __init__(self, storage_path: str = "~/.devorika/memory"):
        self.storage_path = Path(storage_path).expanduser()
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.tasks_file = self.storage_path / "tasks.json"
        self.solutions_file = self.storage_path / "solutions.json"
        self.errors_file = self.storage_path / "errors.json"
        self.preferences_file = self.storage_path / "preferences.json"

        # In-memory caches
        self.tasks: List[Dict] = self._load_json(self.tasks_file)
        self.solutions: Dict[str, Any] = self._load_json(self.solutions_file)
        self.errors: List[Dict] = self._load_json(self.errors_file)
        self.preferences: Dict[str, Any] = self._load_json(self.preferences_file)

    def _load_json(self, file_path: Path) -> Any:
        """Load JSON from file."""
        if file_path.exists():
            try:
                with open(file_path, 'r') as f:
                    return json.load(f)
            except:
                pass
        return [] if file_path.name in ['tasks.json', 'errors.json'] else {}

    def _save_json(self, file_path: Path, data: Any):
        """Save JSON to file."""
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)

    def record_task(self, task: str, result: str, success: bool, metadata: Optional[Dict] = None):
        """Record a completed task."""
        task_record = {
            "timestamp": datetime.now().isoformat(),
            "task": task,
            "result": result,
            "success": success,
            "metadata": metadata or {}
        }
        self.tasks.append(task_record)
        self._save_json(self.tasks_file, self.tasks)

    def record_solution(self, problem_type: str, solution: str, effectiveness: float = 1.0):
        """Record a successful solution pattern."""
        if problem_type not in self.solutions:
            self.solutions[problem_type] = []

        solution_record = {
            "solution": solution,
            "effectiveness": effectiveness,
            "timestamp": datetime.now().isoformat(),
            "use_count": 0
        }
        self.solutions[problem_type].append(solution_record)
        self._save_json(self.solutions_file, self.solutions)

    def get_similar_solutions(self, problem_type: str, limit: int = 5) -> List[Dict]:
        """Get solutions for similar problems."""
        solutions = self.solutions.get(problem_type, [])

        # Sort by effectiveness and use count
        sorted_solutions = sorted(
            solutions,
            key=lambda s: (s['effectiveness'], s['use_count']),
            reverse=True
        )

        return sorted_solutions[:limit]

    def record_error(self, error: str, context: str, solution: Optional[str] = None):
        """Record an error and its solution."""
        error_record = {
            "timestamp": datetime.now().isoformat(),
            "error": error,
            "context": context,
            "solution": solution
        }
        self.errors.append(error_record)
        self._save_json(self.errors_file, self.errors)

    def get_error_solutions(self, error_pattern: str) -> List[Dict]:
        """Get solutions for similar errors."""
        matching_errors = []

        for error_record in self.errors:
            if error_record.get('solution') and error_pattern.lower() in error_record['error'].lower():
                matching_errors.append(error_record)

        return matching_errors[-5:]  # Return last 5 matches

    def set_preference(self, key: str, value: Any):
        """Set a user preference."""
        self.preferences[key] = value
        self._save_json(self.preferences_file, self.preferences)

    def get_preference(self, key: str, default: Any = None) -> Any:
        """Get a user preference."""
        return self.preferences.get(key, default)

    def get_task_history(self, limit: int = 10) -> List[Dict]:
        """Get recent task history."""
        return self.tasks[-limit:]

    def get_statistics(self) -> Dict[str, Any]:
        """Get memory statistics."""
        total_tasks = len(self.tasks)
        successful_tasks = sum(1 for t in self.tasks if t.get('success'))

        return {
            "total_tasks": total_tasks,
            "successful_tasks": successful_tasks,
            "success_rate": (successful_tasks / total_tasks * 100) if total_tasks > 0 else 0,
            "total_solutions": sum(len(sols) for sols in self.solutions.values()),
            "total_errors_recorded": len(self.errors)
        }

    def clear(self):
        """Clear all memory (use with caution)."""
        self.tasks = []
        self.solutions = {}
        self.errors = []
        self.preferences = {}

        self._save_json(self.tasks_file, self.tasks)
        self._save_json(self.solutions_file, self.solutions)
        self._save_json(self.errors_file, self.errors)
        self._save_json(self.preferences_file, self.preferences)
