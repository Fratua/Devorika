"""
DevorikaAgent - The main AI software engineer agent
"""

import json
from typing import List, Dict, Any, Optional
from devorika.core.llm_provider import LLMRouter
from devorika.core.planner import TaskPlanner, TaskStatus
from devorika.core.memory import Memory
from devorika.tools.base import ToolRegistry
from devorika.tools.file_tools import (
    ReadFileTool, WriteFileTool, EditFileTool,
    ListDirectoryTool, SearchCodeTool
)
from devorika.tools.execution_tools import (
    BashTool, PythonExecuteTool, InstallPackageTool, RunTestsTool
)
from devorika.tools.code_analysis import (
    AnalyzeCodeTool, FindBugsTool, GetCodeComplexityTool
)
from devorika.tools.git_tools import (
    GitStatusTool, GitDiffTool, GitCommitTool,
    GitLogTool, GitBranchTool, GitPushTool
)
from devorika.tools.web_tools import (
    WebSearchTool, FetchURLTool, ReadDocumentationTool
)


class DevorikaAgent:
    """
    Main AI software engineer agent.

    Features:
    - Autonomous task execution
    - Advanced planning and decomposition
    - Multi-tool orchestration
    - Learning from past experiences
    - Self-debugging and error recovery
    """

    def __init__(
        self,
        primary_llm: str = "claude",
        fallback_llm: Optional[str] = "gpt",
        memory_enabled: bool = True,
        verbose: bool = True
    ):
        self.verbose = verbose
        self.llm_router = LLMRouter(primary=primary_llm, fallback=fallback_llm)
        self.planner = TaskPlanner(llm_router=self.llm_router)
        self.memory = Memory() if memory_enabled else None
        self.tool_registry = ToolRegistry()

        # Register all tools
        self._register_tools()

        # Conversation history
        self.conversation_history: List[Dict[str, str]] = []

    def _register_tools(self):
        """Register all available tools."""
        # File tools
        self.tool_registry.register(ReadFileTool())
        self.tool_registry.register(WriteFileTool())
        self.tool_registry.register(EditFileTool())
        self.tool_registry.register(ListDirectoryTool())
        self.tool_registry.register(SearchCodeTool())

        # Execution tools
        self.tool_registry.register(BashTool())
        self.tool_registry.register(PythonExecuteTool())
        self.tool_registry.register(InstallPackageTool())
        self.tool_registry.register(RunTestsTool())

        # Code analysis tools
        self.tool_registry.register(AnalyzeCodeTool())
        self.tool_registry.register(FindBugsTool())
        self.tool_registry.register(GetCodeComplexityTool())

        # Git tools
        self.tool_registry.register(GitStatusTool())
        self.tool_registry.register(GitDiffTool())
        self.tool_registry.register(GitCommitTool())
        self.tool_registry.register(GitLogTool())
        self.tool_registry.register(GitBranchTool())
        self.tool_registry.register(GitPushTool())

        # Web tools
        self.tool_registry.register(WebSearchTool())
        self.tool_registry.register(FetchURLTool())
        self.tool_registry.register(ReadDocumentationTool())

    def execute(self, task: str, max_iterations: int = 20) -> str:
        """
        Execute a task autonomously.

        Args:
            task: The task description
            max_iterations: Maximum number of iterations to prevent infinite loops

        Returns:
            Result of the task execution
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"DEVORIKA - Task: {task}")
            print(f"{'='*60}\n")

        # Check memory for similar tasks
        if self.memory:
            similar_solutions = self.memory.get_similar_solutions(self._classify_task(task))
            if similar_solutions and self.verbose:
                print(f"📚 Found {len(similar_solutions)} similar solutions in memory\n")

        # Plan the task
        if self.verbose:
            print("🎯 Planning task...\n")

        tasks = self.planner.decompose_task(task)

        if self.verbose:
            print(self.planner.get_task_summary())
            print()

        # Execute tasks
        iteration = 0
        while iteration < max_iterations:
            iteration += 1

            # Get next task
            next_task = self.planner.get_next_task()
            if not next_task:
                # Check if all tasks are completed
                progress = self.planner.get_progress()
                if progress['completed'] == progress['total']:
                    if self.verbose:
                        print("\n✅ All tasks completed successfully!")
                    break
                else:
                    if self.verbose:
                        print("\n⚠️ No more tasks available but some are incomplete")
                    break

            # Execute the task
            if self.verbose:
                print(f"\n🔄 Executing: {next_task.description}")

            self.planner.update_task_status(next_task.id, TaskStatus.IN_PROGRESS)

            try:
                result = self._execute_single_task(next_task.description)
                self.planner.update_task_status(next_task.id, TaskStatus.COMPLETED, result=result)

                if self.verbose:
                    print(f"✅ Completed: {next_task.description}")
                    if result and len(result) < 200:
                        print(f"   Result: {result}")

            except Exception as e:
                error_msg = str(e)
                self.planner.update_task_status(next_task.id, TaskStatus.FAILED, error=error_msg)

                if self.verbose:
                    print(f"❌ Failed: {next_task.description}")
                    print(f"   Error: {error_msg}")

                # Record error in memory
                if self.memory:
                    self.memory.record_error(error_msg, next_task.description)

                # Try to recover
                if self.verbose:
                    print("   🔧 Attempting recovery...")

                try:
                    recovery_result = self._recover_from_error(next_task.description, error_msg)
                    self.planner.update_task_status(next_task.id, TaskStatus.COMPLETED, result=recovery_result)
                    if self.verbose:
                        print("   ✅ Recovered successfully!")
                except Exception as recovery_error:
                    if self.verbose:
                        print(f"   ❌ Recovery failed: {recovery_error}")
                    break

        # Get final result
        progress = self.planner.get_progress()
        success = progress['completed'] == progress['total']

        final_result = self.planner.get_task_summary()

        # Record in memory
        if self.memory:
            self.memory.record_task(
                task=task,
                result=final_result,
                success=success,
                metadata={"iterations": iteration, "progress": progress}
            )

        return final_result

    def _execute_single_task(self, task_description: str) -> str:
        """Execute a single task using the LLM and tools."""
        # Build context with available tools
        tools_description = self._get_tools_description()

        prompt = f"""You are Devorika, an advanced AI software engineer. Execute this task:

Task: {task_description}

Available tools:
{tools_description}

Think step by step:
1. Analyze what needs to be done
2. Determine which tools to use
3. Execute the tools in the right order
4. Verify the result

Respond with a JSON object:
{{
    "reasoning": "Your thought process",
    "actions": [
        {{"tool": "tool_name", "parameters": {{"param": "value"}}}},
        ...
    ],
    "expected_outcome": "What you expect to happen"
}}
"""

        # Get LLM response
        messages = self.conversation_history + [{"role": "user", "content": prompt}]
        response = self.llm_router.generate(messages, task_type="code_generation")

        self.conversation_history.append({"role": "user", "content": prompt})
        self.conversation_history.append({"role": "assistant", "content": response})

        # Parse and execute actions
        try:
            # Extract JSON from response
            start = response.find('{')
            end = response.rfind('}') + 1
            if start != -1 and end != 0:
                action_plan = json.loads(response[start:end])

                if self.verbose and 'reasoning' in action_plan:
                    print(f"   💭 Reasoning: {action_plan['reasoning']}")

                results = []
                for action in action_plan.get('actions', []):
                    tool_name = action.get('tool')
                    params = action.get('parameters', {})

                    if self.verbose:
                        print(f"   🔧 Using tool: {tool_name}")

                    result = self.tool_registry.execute(tool_name, **params)
                    results.append(result)

                    if self.verbose and isinstance(result, str) and len(result) < 200:
                        print(f"      Result: {result}")

                return " | ".join(str(r) for r in results)
            else:
                # If no JSON, return the response as-is
                return response

        except Exception as e:
            if self.verbose:
                print(f"   ⚠️ Error executing action plan: {e}")
            # Return response even if parsing fails
            return response

    def _recover_from_error(self, task: str, error: str) -> str:
        """Attempt to recover from an error using self-debugging."""
        # Check memory for similar errors
        if self.memory:
            similar_errors = self.memory.get_error_solutions(error)
            if similar_errors:
                context = f"Similar errors in memory: {json.dumps(similar_errors, indent=2)}"
            else:
                context = "No similar errors in memory"
        else:
            context = ""

        recovery_prompt = f"""An error occurred while executing this task:

Task: {task}
Error: {error}

{context}

Analyze the error and provide a solution. Respond with a JSON object:
{{
    "diagnosis": "What went wrong",
    "solution": "How to fix it",
    "actions": [
        {{"tool": "tool_name", "parameters": {{"param": "value"}}}}
    ]
}}
"""

        messages = [{"role": "user", "content": recovery_prompt}]
        response = self.llm_router.generate(messages, task_type="debugging")

        # Execute recovery actions
        try:
            start = response.find('{')
            end = response.rfind('}') + 1
            if start != -1 and end != 0:
                recovery_plan = json.loads(response[start:end])

                results = []
                for action in recovery_plan.get('actions', []):
                    tool_name = action.get('tool')
                    params = action.get('parameters', {})
                    result = self.tool_registry.execute(tool_name, **params)
                    results.append(result)

                # Record the solution
                if self.memory:
                    self.memory.record_error(error, task, solution=recovery_plan.get('solution'))

                return " | ".join(str(r) for r in results)
        except Exception as e:
            raise Exception(f"Recovery failed: {e}")

    def _get_tools_description(self) -> str:
        """Get formatted description of available tools."""
        tools = self.tool_registry.list_tools()
        descriptions = []
        for tool in tools:
            descriptions.append(f"- {tool['name']}: {tool['description']}")
        return "\n".join(descriptions)

    def _classify_task(self, task: str) -> str:
        """Classify task type for memory lookup."""
        task_lower = task.lower()
        if any(word in task_lower for word in ['implement', 'create', 'add', 'build']):
            return "implementation"
        elif any(word in task_lower for word in ['debug', 'fix', 'error', 'bug']):
            return "debugging"
        elif any(word in task_lower for word in ['refactor', 'optimize', 'improve']):
            return "refactoring"
        elif any(word in task_lower for word in ['test', 'verify']):
            return "testing"
        else:
            return "general"

    def chat(self, message: str) -> str:
        """Have a conversation with Devorika."""
        self.conversation_history.append({"role": "user", "content": message})

        response = self.llm_router.generate(
            self.conversation_history,
            task_type="general"
        )

        self.conversation_history.append({"role": "assistant", "content": response})

        return response
