"""Entry point for running the function-calling agent."""

from func_agent import FunctionCallingAgent
import functions


def main():
    """Build the agent, register tools, and execute the prompt batch."""
    agent = FunctionCallingAgent()
    agent.register_tools([
        functions.fn_add_numbers,
        functions.fn_subtract_numbers,
        functions.fn_multiply_numbers,
        functions.fn_greet,
        functions.fn_reverse_string,
        functions.get_current_weather,
        functions.fn_divide,
    ])

    agent.run()


if __name__ == "__main__":
    main()
