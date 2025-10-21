"""
Specialized agents for specific software engineering tasks
"""

from devorika.core.agent import DevorikaAgent


class CodeGenerationAgent(DevorikaAgent):
    """Specialized agent for code generation tasks."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.specialization = "code_generation"

    def execute(self, task: str, max_iterations: int = 20) -> str:
        """Execute code generation task with specialized approach."""
        # Add code generation specific context
        enhanced_task = f"""As a code generation specialist, {task}

Focus on:
- Writing clean, efficient, and well-structured code
- Following best practices and design patterns
- Adding appropriate error handling
- Writing self-documenting code with clear variable names
"""
        return super().execute(enhanced_task, max_iterations)


class DebuggingAgent(DevorikaAgent):
    """Specialized agent for debugging and fixing issues."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.specialization = "debugging"

    def execute(self, task: str, max_iterations: int = 20) -> str:
        """Execute debugging task with specialized approach."""
        enhanced_task = f"""As a debugging specialist, {task}

Systematic debugging approach:
1. Reproduce the issue
2. Gather relevant information (logs, stack traces, state)
3. Form hypotheses about the root cause
4. Test each hypothesis
5. Implement the fix
6. Verify the fix works and doesn't break anything else
"""
        return super().execute(enhanced_task, max_iterations)


class TestingAgent(DevorikaAgent):
    """Specialized agent for writing and running tests."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.specialization = "testing"

    def execute(self, task: str, max_iterations: int = 20) -> str:
        """Execute testing task with specialized approach."""
        enhanced_task = f"""As a testing specialist, {task}

Testing strategy:
1. Identify what needs to be tested (units, integration, edge cases)
2. Write comprehensive test cases
3. Ensure good code coverage
4. Test both success and failure scenarios
5. Include edge cases and boundary conditions
6. Make tests clear and maintainable
"""
        return super().execute(enhanced_task, max_iterations)


class DocumentationAgent(DevorikaAgent):
    """Specialized agent for writing documentation."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.specialization = "documentation"

    def execute(self, task: str, max_iterations: int = 20) -> str:
        """Execute documentation task with specialized approach."""
        enhanced_task = f"""As a documentation specialist, {task}

Documentation best practices:
1. Start with a clear overview
2. Explain the "why" not just the "what"
3. Include usage examples
4. Document parameters, return values, and exceptions
5. Add diagrams or code examples where helpful
6. Keep it concise but complete
"""
        return super().execute(enhanced_task, max_iterations)


class CodeReviewAgent(DevorikaAgent):
    """Specialized agent for code review."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.specialization = "code_review"

    def execute(self, task: str, max_iterations: int = 20) -> str:
        """Execute code review task with specialized approach."""
        enhanced_task = f"""As a code review specialist, {task}

Review checklist:
1. Code quality and readability
2. Adherence to best practices and patterns
3. Potential bugs or edge cases
4. Performance considerations
5. Security issues
6. Test coverage
7. Documentation completeness
8. Suggest improvements

Provide constructive feedback with specific examples.
"""
        return super().execute(enhanced_task, max_iterations)


class RefactoringAgent(DevorikaAgent):
    """Specialized agent for code refactoring."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.specialization = "refactoring"

    def execute(self, task: str, max_iterations: int = 20) -> str:
        """Execute refactoring task with specialized approach."""
        enhanced_task = f"""As a refactoring specialist, {task}

Refactoring approach:
1. Understand current implementation thoroughly
2. Ensure tests exist before refactoring
3. Make incremental changes
4. Run tests after each change
5. Improve code structure and readability
6. Reduce duplication and complexity
7. Maintain or improve performance
"""
        return super().execute(enhanced_task, max_iterations)


class ArchitectureAgent(DevorikaAgent):
    """Specialized agent for software architecture and design."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.specialization = "architecture"

    def execute(self, task: str, max_iterations: int = 20) -> str:
        """Execute architecture task with specialized approach."""
        enhanced_task = f"""As a software architecture specialist, {task}

Architecture considerations:
1. Identify key components and their responsibilities
2. Define clear interfaces and contracts
3. Consider scalability and maintainability
4. Choose appropriate design patterns
5. Plan for extensibility
6. Document architectural decisions
"""
        return super().execute(enhanced_task, max_iterations)
