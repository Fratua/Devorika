#!/usr/bin/env python3
"""
Devorika CLI - Command-line interface for the AI software programmer
"""

import sys
import argparse
from typing import Optional
from devorika.core.agent import DevorikaAgent
from devorika.core.orchestrator import Orchestrator
from devorika.core.memory import Memory


class DevorikaCLI:
    """Command-line interface for Devorika."""

    def __init__(self):
        self.agent: Optional[DevorikaAgent] = None
        self.orchestrator: Optional[Orchestrator] = None

    def run(self, args):
        """Run the CLI with given arguments."""
        if args.command == "execute":
            self._execute_task(args)
        elif args.command == "chat":
            self._chat_mode(args)
        elif args.command == "pipeline":
            self._pipeline_mode(args)
        elif args.command == "parallel":
            self._parallel_mode(args)
        elif args.command == "memory":
            self._memory_command(args)
        elif args.command == "version":
            self._show_version()
        else:
            print("Unknown command. Use --help for usage information.")

    def _execute_task(self, args):
        """Execute a single task."""
        print(f"\n{'='*60}")
        print("DEVORIKA - Advanced AI Software Programmer")
        print(f"{'='*60}\n")

        self.agent = DevorikaAgent(
            primary_llm=args.llm,
            verbose=args.verbose,
            memory_enabled=not args.no_memory
        )

        result = self.agent.execute(args.task, max_iterations=args.max_iterations)

        print("\n" + "="*60)
        print("FINAL RESULT")
        print("="*60)
        print(result)

    def _chat_mode(self, args):
        """Interactive chat mode."""
        print(f"\n{'='*60}")
        print("DEVORIKA - Chat Mode")
        print("Type 'exit' or 'quit' to end the conversation")
        print(f"{'='*60}\n")

        self.agent = DevorikaAgent(
            primary_llm=args.llm,
            verbose=args.verbose,
            memory_enabled=not args.no_memory
        )

        while True:
            try:
                user_input = input("\nYou: ").strip()

                if user_input.lower() in ['exit', 'quit', 'bye']:
                    print("\nDevorika: Goodbye! Happy coding! 👋")
                    break

                if not user_input:
                    continue

                # Check if it's a task execution request
                if user_input.startswith("/execute "):
                    task = user_input[9:]
                    result = self.agent.execute(task)
                    print(f"\nDevorika:\n{result}")
                else:
                    # Regular chat
                    response = self.agent.chat(user_input)
                    print(f"\nDevorika: {response}")

            except KeyboardInterrupt:
                print("\n\nDevorika: Goodbye! 👋")
                break
            except Exception as e:
                print(f"\nError: {e}")

    def _pipeline_mode(self, args):
        """Execute task through a quality pipeline."""
        print(f"\n{'='*60}")
        print("DEVORIKA - Pipeline Mode")
        print(f"{'='*60}\n")

        self.orchestrator = Orchestrator(
            max_workers=args.workers,
            verbose=args.verbose
        )

        result = self.orchestrator.execute_pipeline(args.task)

        print("\n" + "="*60)
        print("PIPELINE RESULT")
        print("="*60)
        print(result)

    def _parallel_mode(self, args):
        """Execute multiple tasks in parallel."""
        print(f"\n{'='*60}")
        print("DEVORIKA - Parallel Execution Mode")
        print(f"{'='*60}\n")

        self.orchestrator = Orchestrator(
            max_workers=args.workers,
            verbose=args.verbose
        )

        # Parse tasks from input
        tasks = []
        for i, task_desc in enumerate(args.tasks):
            task_type = "general"
            if ":" in task_desc:
                task_type, task_desc = task_desc.split(":", 1)

            tasks.append({
                "type": task_type.strip(),
                "description": task_desc.strip()
            })

        results = self.orchestrator.execute_parallel(tasks)

        print("\n" + "="*60)
        print("PARALLEL EXECUTION RESULTS")
        print("="*60)
        for i, result in sorted(results.items()):
            print(f"\nTask {i}: {result['task']['description']}")
            print(f"Status: {result['status']}")
            print(f"Result: {result['result'][:200]}...")
            print("-" * 60)

    def _memory_command(self, args):
        """Memory management commands."""
        memory = Memory()

        if args.memory_action == "stats":
            stats = memory.get_statistics()
            print("\n" + "="*60)
            print("MEMORY STATISTICS")
            print("="*60)
            for key, value in stats.items():
                print(f"{key}: {value}")

        elif args.memory_action == "history":
            history = memory.get_task_history(limit=args.limit or 10)
            print("\n" + "="*60)
            print("TASK HISTORY")
            print("="*60)
            for task in history:
                print(f"\nTimestamp: {task['timestamp']}")
                print(f"Task: {task['task']}")
                print(f"Success: {task['success']}")
                print("-" * 60)

        elif args.memory_action == "clear":
            confirm = input("Are you sure you want to clear all memory? (yes/no): ")
            if confirm.lower() == "yes":
                memory.clear()
                print("Memory cleared successfully.")
            else:
                print("Operation cancelled.")

    def _show_version(self):
        """Show version information."""
        from devorika import __version__
        print(f"\nDevorika v{__version__}")
        print("Advanced AI Software Programmer")
        print("\nSuperior to Devin with:")
        print("  ✓ Multi-LLM support (Claude, GPT, local models)")
        print("  ✓ Multi-agent collaboration")
        print("  ✓ Advanced planning and decomposition")
        print("  ✓ Long-term memory and learning")
        print("  ✓ Parallel task execution")
        print("  ✓ Self-debugging capabilities")
        print("  ✓ Extensible plugin architecture\n")


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Devorika - Advanced AI Software Programmer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  devorika execute "Create a Python web scraper"
  devorika chat
  devorika pipeline "Build a REST API with authentication"
  devorika parallel "coder:Implement user login" "tester:Write tests for login"
  devorika memory stats
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Execute command
    execute_parser = subparsers.add_parser("execute", help="Execute a single task")
    execute_parser.add_argument("task", help="Task description")
    execute_parser.add_argument("--llm", default="claude", help="LLM to use (default: claude)")
    execute_parser.add_argument("--max-iterations", type=int, default=20, help="Max iterations")
    execute_parser.add_argument("--no-memory", action="store_true", help="Disable memory")
    execute_parser.add_argument("--verbose", action="store_true", default=True, help="Verbose output")

    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Interactive chat mode")
    chat_parser.add_argument("--llm", default="claude", help="LLM to use (default: claude)")
    chat_parser.add_argument("--no-memory", action="store_true", help="Disable memory")
    chat_parser.add_argument("--verbose", action="store_true", default=True, help="Verbose output")

    # Pipeline command
    pipeline_parser = subparsers.add_parser("pipeline", help="Execute through quality pipeline")
    pipeline_parser.add_argument("task", help="Task description")
    pipeline_parser.add_argument("--workers", type=int, default=4, help="Max parallel workers")
    pipeline_parser.add_argument("--verbose", action="store_true", default=True, help="Verbose output")

    # Parallel command
    parallel_parser = subparsers.add_parser("parallel", help="Execute multiple tasks in parallel")
    parallel_parser.add_argument("tasks", nargs="+", help="Tasks to execute (format: [type:]description)")
    parallel_parser.add_argument("--workers", type=int, default=4, help="Max parallel workers")
    parallel_parser.add_argument("--verbose", action="store_true", default=True, help="Verbose output")

    # Memory command
    memory_parser = subparsers.add_parser("memory", help="Manage memory")
    memory_parser.add_argument("memory_action", choices=["stats", "history", "clear"], help="Memory action")
    memory_parser.add_argument("--limit", type=int, help="Limit for history command")

    # Version command
    subparsers.add_parser("version", help="Show version information")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    cli = DevorikaCLI()
    try:
        cli.run(args)
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        if args.command in ["execute", "chat", "pipeline", "parallel"]:
            print("\nPlease ensure you have set up your API keys:")
            print("  export ANTHROPIC_API_KEY='your-key-here'")
            print("  export OPENAI_API_KEY='your-key-here'")
        sys.exit(1)


if __name__ == "__main__":
    main()
