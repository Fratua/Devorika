"""
Multi-Agent Orchestrator
Manages collaboration between multiple specialized agents
"""

from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from devorika.core.agent import DevorikaAgent
from devorika.agents.specialist_agents import (
    CodeGenerationAgent,
    DebuggingAgent,
    TestingAgent,
    DocumentationAgent,
    CodeReviewAgent
)


class Orchestrator:
    """
    Orchestrates multiple specialized agents for complex tasks.

    This is a key advantage over Devin - parallel execution with specialized agents.
    """

    def __init__(self, max_workers: int = 4, verbose: bool = True):
        self.max_workers = max_workers
        self.verbose = verbose

        # Initialize specialized agents
        self.agents = {
            "general": DevorikaAgent(verbose=verbose),
            "coder": CodeGenerationAgent(verbose=verbose),
            "debugger": DebuggingAgent(verbose=verbose),
            "tester": TestingAgent(verbose=verbose),
            "documenter": DocumentationAgent(verbose=verbose),
            "reviewer": CodeReviewAgent(verbose=verbose)
        }

    def execute_parallel(self, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Execute multiple tasks in parallel using specialized agents.

        Args:
            tasks: List of task dictionaries with 'type' and 'description'

        Returns:
            Dictionary of results keyed by task index
        """
        if self.verbose:
            print(f"\n🚀 Orchestrator: Executing {len(tasks)} tasks in parallel\n")

        results = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_task = {}
            for i, task in enumerate(tasks):
                agent_type = task.get('type', 'general')
                agent = self.agents.get(agent_type, self.agents['general'])

                future = executor.submit(agent.execute, task['description'])
                future_to_task[future] = (i, task)

            # Collect results as they complete
            for future in as_completed(future_to_task):
                i, task = future_to_task[future]
                try:
                    result = future.result()
                    results[i] = {
                        "task": task,
                        "result": result,
                        "status": "success"
                    }
                    if self.verbose:
                        print(f"✅ Task {i} completed: {task['description'][:50]}...")
                except Exception as e:
                    results[i] = {
                        "task": task,
                        "result": str(e),
                        "status": "failed"
                    }
                    if self.verbose:
                        print(f"❌ Task {i} failed: {task['description'][:50]}...")

        if self.verbose:
            print(f"\n✨ Orchestrator: All tasks completed\n")

        return results

    def execute_pipeline(self, task: str) -> str:
        """
        Execute a task through a pipeline of specialized agents.

        Pipeline: Planning -> Implementation -> Testing -> Review -> Documentation

        This sequential approach ensures quality at each stage.
        """
        if self.verbose:
            print(f"\n🔄 Orchestrator: Executing pipeline for task: {task}\n")

        results = []

        # Step 1: Plan with general agent
        if self.verbose:
            print("📋 Stage 1: Planning...")
        planning_result = self.agents['general'].execute(f"Plan how to {task}")
        results.append(("Planning", planning_result))

        # Step 2: Implement with coder agent
        if self.verbose:
            print("\n💻 Stage 2: Implementation...")
        implementation_result = self.agents['coder'].execute(task)
        results.append(("Implementation", implementation_result))

        # Step 3: Test with tester agent
        if self.verbose:
            print("\n🧪 Stage 3: Testing...")
        testing_result = self.agents['tester'].execute(f"Create and run tests for: {task}")
        results.append(("Testing", testing_result))

        # Step 4: Review with reviewer agent
        if self.verbose:
            print("\n🔍 Stage 4: Code Review...")
        review_result = self.agents['reviewer'].execute(f"Review the implementation of: {task}")
        results.append(("Review", review_result))

        # Step 5: Document with documenter agent
        if self.verbose:
            print("\n📝 Stage 5: Documentation...")
        documentation_result = self.agents['documenter'].execute(f"Document: {task}")
        results.append(("Documentation", documentation_result))

        # Compile final report
        final_report = "="*60 + "\n"
        final_report += "PIPELINE EXECUTION COMPLETE\n"
        final_report += "="*60 + "\n\n"

        for stage, result in results:
            final_report += f"{stage}:\n{'-'*60}\n"
            # Truncate long results
            if len(result) > 500:
                final_report += result[:500] + "\n[...truncated...]\n"
            else:
                final_report += result + "\n"
            final_report += "\n"

        if self.verbose:
            print("\n✨ Orchestrator: Pipeline completed successfully!\n")

        return final_report

    def collaborate(self, task: str, agent_types: List[str]) -> str:
        """
        Have multiple agents collaborate on a single task.

        Each agent contributes their perspective, and results are synthesized.
        """
        if self.verbose:
            print(f"\n🤝 Orchestrator: Collaborative execution with {len(agent_types)} agents\n")

        contributions = []

        for agent_type in agent_types:
            agent = self.agents.get(agent_type)
            if not agent:
                if self.verbose:
                    print(f"⚠️ Agent type '{agent_type}' not found, skipping")
                continue

            if self.verbose:
                print(f"   Agent '{agent_type}' contributing...")

            contribution = agent.execute(task)
            contributions.append({
                "agent": agent_type,
                "contribution": contribution
            })

        # Synthesize contributions using general agent
        synthesis_prompt = f"Synthesize these contributions into a cohesive solution:\n\n"
        for contrib in contributions:
            synthesis_prompt += f"{contrib['agent']} says:\n{contrib['contribution']}\n\n"

        synthesis = self.agents['general'].chat(synthesis_prompt)

        if self.verbose:
            print("\n✅ Orchestrator: Collaboration complete\n")

        return synthesis

    def get_agent(self, agent_type: str) -> Optional[DevorikaAgent]:
        """Get a specific agent."""
        return self.agents.get(agent_type)

    def add_agent(self, agent_type: str, agent: DevorikaAgent):
        """Add a custom agent to the orchestrator."""
        self.agents[agent_type] = agent
