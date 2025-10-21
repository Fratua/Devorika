"""
Advanced Planning and Task Decomposition System
Breaks down complex tasks into manageable steps
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum


class TaskStatus(Enum):
    """Task status enumeration."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


@dataclass
class Task:
    """Represents a single task."""
    id: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    dependencies: List[str] = None
    priority: int = 0
    result: Optional[str] = None
    error: Optional[str] = None

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


class TaskPlanner:
    """Advanced task planning and decomposition system."""

    def __init__(self, llm_router=None):
        self.llm_router = llm_router
        self.tasks: Dict[str, Task] = {}
        self.task_counter = 0

    def decompose_task(self, goal: str, context: Optional[str] = None) -> List[Task]:
        """
        Decompose a high-level goal into specific tasks.

        Uses LLM to intelligently break down complex goals.
        """
        if not self.llm_router:
            # Fallback to basic decomposition
            return self._basic_decomposition(goal)

        # Use LLM for intelligent decomposition
        prompt = f"""You are an expert software engineer. Break down this goal into specific, actionable tasks.

Goal: {goal}

{f"Context: {context}" if context else ""}

Provide a list of tasks in JSON format:
[
  {{"id": "1", "description": "Task description", "dependencies": [], "priority": 1}},
  ...
]

Tasks should be:
1. Specific and actionable
2. Ordered logically with dependencies
3. Prioritized appropriately
4. Comprehensive but not overly granular
"""

        try:
            response = self.llm_router.generate(
                messages=[{"role": "user", "content": prompt}],
                task_type="planning"
            )

            # Parse JSON response
            import json
            # Extract JSON from response
            start = response.find('[')
            end = response.rfind(']') + 1
            if start != -1 and end != 0:
                task_data = json.loads(response[start:end])
                tasks = []
                for t in task_data:
                    task = Task(
                        id=t["id"],
                        description=t["description"],
                        dependencies=t.get("dependencies", []),
                        priority=t.get("priority", 0)
                    )
                    tasks.append(task)
                    self.tasks[task.id] = task
                return tasks
        except Exception as e:
            print(f"LLM decomposition failed: {e}, using basic decomposition")

        return self._basic_decomposition(goal)

    def _basic_decomposition(self, goal: str) -> List[Task]:
        """Basic rule-based task decomposition."""
        tasks = []

        # Common software engineering task patterns
        if "implement" in goal.lower() or "create" in goal.lower():
            steps = [
                "Analyze requirements and understand the goal",
                "Design the solution architecture",
                "Implement core functionality",
                "Add error handling and validation",
                "Write tests",
                "Document the implementation"
            ]
        elif "debug" in goal.lower() or "fix" in goal.lower():
            steps = [
                "Reproduce the issue",
                "Analyze the code and identify the root cause",
                "Develop a fix",
                "Test the fix",
                "Verify no regressions"
            ]
        elif "refactor" in goal.lower():
            steps = [
                "Understand current implementation",
                "Identify refactoring opportunities",
                "Plan refactoring approach",
                "Refactor code incrementally",
                "Ensure tests pass",
                "Update documentation"
            ]
        else:
            steps = [
                "Understand the requirements",
                "Plan the approach",
                "Execute the plan",
                "Verify the results"
            ]

        for i, step in enumerate(steps):
            task = Task(
                id=str(i + 1),
                description=step,
                priority=len(steps) - i
            )
            tasks.append(task)
            self.tasks[task.id] = task

        return tasks

    def get_next_task(self) -> Optional[Task]:
        """Get the next task to execute based on dependencies and priority."""
        available_tasks = []

        for task in self.tasks.values():
            if task.status != TaskStatus.PENDING:
                continue

            # Check if all dependencies are completed
            dependencies_met = all(
                self.tasks.get(dep_id, Task(id="", description="")).status == TaskStatus.COMPLETED
                for dep_id in task.dependencies
            )

            if dependencies_met:
                available_tasks.append(task)

        if not available_tasks:
            return None

        # Return highest priority task
        return max(available_tasks, key=lambda t: t.priority)

    def update_task_status(self, task_id: str, status: TaskStatus, result: Optional[str] = None, error: Optional[str] = None):
        """Update task status."""
        if task_id in self.tasks:
            self.tasks[task_id].status = status
            if result:
                self.tasks[task_id].result = result
            if error:
                self.tasks[task_id].error = error

    def get_progress(self) -> Dict[str, Any]:
        """Get current progress statistics."""
        total = len(self.tasks)
        completed = sum(1 for t in self.tasks.values() if t.status == TaskStatus.COMPLETED)
        failed = sum(1 for t in self.tasks.values() if t.status == TaskStatus.FAILED)
        in_progress = sum(1 for t in self.tasks.values() if t.status == TaskStatus.IN_PROGRESS)

        return {
            "total": total,
            "completed": completed,
            "failed": failed,
            "in_progress": in_progress,
            "percentage": (completed / total * 100) if total > 0 else 0
        }

    def get_task_summary(self) -> str:
        """Get a formatted summary of all tasks."""
        output = "Task Summary:\n\n"

        for task in sorted(self.tasks.values(), key=lambda t: t.id):
            status_icon = {
                TaskStatus.PENDING: "⏳",
                TaskStatus.IN_PROGRESS: "🔄",
                TaskStatus.COMPLETED: "✅",
                TaskStatus.FAILED: "❌",
                TaskStatus.BLOCKED: "🚫"
            }.get(task.status, "❓")

            output += f"{status_icon} [{task.id}] {task.description}\n"
            if task.dependencies:
                output += f"    Dependencies: {', '.join(task.dependencies)}\n"
            if task.error:
                output += f"    Error: {task.error}\n"

        progress = self.get_progress()
        output += f"\nProgress: {progress['completed']}/{progress['total']} ({progress['percentage']:.1f}%)"

        return output
