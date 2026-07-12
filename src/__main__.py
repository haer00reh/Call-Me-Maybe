"""Entry point for running the function-calling agent."""

from .func_agent import FunctionCallingAgent


def main() -> None:
    """Build the agent, register tools, and execute the prompt batch."""
    agent = FunctionCallingAgent()

    agent.run()


if __name__ == "__main__":
    main()
