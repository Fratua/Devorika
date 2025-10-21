"""
Basic Usage Examples for Devorika
"""

from devorika import DevorikaAgent, Orchestrator


def example_1_simple_task():
    """Execute a simple task."""
    print("Example 1: Simple Task Execution\n")

    agent = DevorikaAgent(verbose=True)
    result = agent.execute("Create a Python function to calculate fibonacci numbers")

    print(f"\nResult:\n{result}")


def example_2_chat_mode():
    """Interactive chat with Devorika."""
    print("Example 2: Chat Mode\n")

    agent = DevorikaAgent(verbose=False)

    # Ask questions
    response1 = agent.chat("What are the best practices for Python error handling?")
    print(f"Response: {response1}\n")

    # Follow-up questions maintain context
    response2 = agent.chat("Can you show me an example?")
    print(f"Follow-up: {response2}\n")


def example_3_parallel_execution():
    """Execute multiple tasks in parallel."""
    print("Example 3: Parallel Execution\n")

    orchestrator = Orchestrator(max_workers=3, verbose=True)

    tasks = [
        {"type": "coder", "description": "Create a user authentication module"},
        {"type": "tester", "description": "Write unit tests for authentication"},
        {"type": "documenter", "description": "Document the authentication API"}
    ]

    results = orchestrator.execute_parallel(tasks)

    for task_id, result in results.items():
        print(f"\nTask {task_id}: {result['status']}")
        print(f"Description: {result['task']['description']}")


def example_4_quality_pipeline():
    """Execute through quality pipeline."""
    print("Example 4: Quality Pipeline\n")

    orchestrator = Orchestrator(verbose=True)

    result = orchestrator.execute_pipeline(
        "Create a RESTful API endpoint for user registration"
    )

    print(f"\nPipeline Result:\n{result}")


def example_5_memory_system():
    """Use the memory system."""
    print("Example 5: Memory System\n")

    from devorika.core.memory import Memory

    memory = Memory()

    # Record a solution
    memory.record_solution(
        problem_type="database_optimization",
        solution="Add indexes to frequently queried columns",
        effectiveness=0.95
    )

    # Retrieve similar solutions
    solutions = memory.get_similar_solutions("database_optimization")
    print(f"Found {len(solutions)} solutions for database optimization")

    # Get statistics
    stats = memory.get_statistics()
    print(f"\nMemory Statistics:")
    print(f"  Total tasks: {stats['total_tasks']}")
    print(f"  Success rate: {stats['success_rate']:.1f}%")


def example_6_custom_agent():
    """Create a custom specialized agent."""
    print("Example 6: Custom Agent\n")

    class SecurityAgent(DevorikaAgent):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.specialization = "security"

        def execute(self, task: str, **kwargs):
            enhanced_task = f"""As a security specialist, {task}

            Focus on:
            - Security vulnerabilities
            - Best security practices
            - Common attack vectors
            - Secure coding patterns
            """
            return super().execute(enhanced_task, **kwargs)

    # Use custom agent
    security_agent = SecurityAgent(verbose=True)
    result = security_agent.execute("Review this authentication code for security issues")

    print(f"\nSecurity Review:\n{result}")


if __name__ == "__main__":
    print("="*60)
    print("DEVORIKA - Usage Examples")
    print("="*60 + "\n")

    # Run examples
    # Uncomment the examples you want to run

    # example_1_simple_task()
    # example_2_chat_mode()
    # example_3_parallel_execution()
    # example_4_quality_pipeline()
    # example_5_memory_system()
    # example_6_custom_agent()

    print("\nNote: Make sure to set your API keys in .env before running!")
    print("  ANTHROPIC_API_KEY=your_key_here")
    print("  OPENAI_API_KEY=your_key_here")
