# AI Interface for DataFrame Transformation

This project provides a simple interface, AIInterface, that uses OpenAI models to generate and execute commands on a pandas DataFrame. It dynamically discovers functions (commands) from external Python modules and uses them to transform dataframes based on user instructions in natural language.

## Features

- Automatic Command Registration: Point the interface at your modules, and it automatically extracts functions as commands.
- OpenAI Integration: Sends both system prompts (command descriptions) and user requests to an OpenAI model for command generation.
- Adaptive Error Handling: If your commands crash and burn, AIInterface politely begs GPT to come up with a better plan.
