import json
import inspect
from typing import Optional, Callable, Any

from openai import OpenAI, OpenAIError
from pandas import DataFrame

from src.interface.prompts import PROMPT_TEMPLATE, PROMPT_TEMPLATE_ERROR

class AIInterface:
    """
    An interface that manages commands, constructs prompts, and communicates
    with OpenAI's API to generate and apply commands on a DataFrame.
    """
    PROMPT_TEMPLATE: str = PROMPT_TEMPLATE
    PROMPT_TEMPLATE_ERROR: str = PROMPT_TEMPLATE_ERROR

    def __init__(self, load_modules: Optional[list[object]] = None):
        self._client = OpenAI()
        #COMMANDS
        self.commands_description: dict[str, str] = {}
        self.commands: dict[str, Callable] = {}

        #primitive cashing
        self._commands_updated = False
        self._available_commands: str = ""

        if load_modules:
            self.add_commands_from_modules(load_modules)


    def add_command(self, name: str, func: Callable) -> None:
        """
        Add a single command to the interface.

        :param name: The name used to reference the command.
        :param func: The function that will be invoked when the command is called.
        """
        self._commands_updated = True
        print(f"Func found - {name}")
        signature = inspect.signature(func)
        docstring = inspect.getdoc(func)
        description = f"{signature} - {docstring if docstring else 'No description'}"
        self.commands_description[name] = description
        self.commands[name] = func

    def add_commands_from_modules(self, modules: list[object]) -> None:
        """
        Add all functions from the given modules as commands.

        :param modules: A list of module objects that contain functions to add.
        """
        for module in modules:
            for name, func in inspect.getmembers(module, inspect.isfunction):
                self.add_command(name, func)

    def _get_available_commands(self) -> str:
        """
        Build a string describing all available commands, if there have been changes
        since the last call.

        :return: A string listing all commands and their descriptions.
        """
        if self._commands_updated and self.commands:
            self._available_commands =  "\n".join(
                    [
                        f" - {name}({description.replace("\n","")})"
                        for name, description in self.commands_description.items()
                    ]
            )
            self._commands_updated = False
        return self._available_commands

    def _get_prompt(self) -> str:
        """
        Construct the initial prompt (system message) for the model, listing all commands.

        :return: The prompt string with placeholders filled in.
        :raises ValueError: If no commands are available.
        """
        if not self.commands:
            raise ValueError("No commands available")
        return self.PROMPT_TEMPLATE.format(
                commands=self._get_available_commands(),
        )

    def _get_error_prompt(self,
            error_message: str,
    ) -> str:
        """
        Construct a follow-up prompt (system message) with an error message for the model to handle.

        :param error_message: The error message to provide to the model.
        :return: The prompt string with the error message filled in.
        """
        return self.PROMPT_TEMPLATE_ERROR.format(
                error_msg=error_message
        )


    def generate_commands(self,
            prompt: str,
            user_request: str,
            engine: str = "gpt-4",
            max_tokens: int = 1000,
            temperature: float = 0.7,
            top_p: float = 1,
    ) -> dict[str, list[dict]]:
        """
        Send a prompt and user request to the OpenAI API to generate a set of commands.

        :param prompt: The system prompt describing available commands.
        :param user_request: The user's direct request to the model.
        :param engine: The model/engine to use (default 'o1').
        :param max_tokens: The maximum number of tokens in the response.
        :param temperature: The sampling temperature.
        :param top_p: The nucleus sampling parameter.
        :return: A dictionary containing a list of generated commands.
        :raises ValueError: If the response from the model is invalid or empty.
        """
        try:
            response = self._client.chat.completions.create(
                    model=engine,
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role":"user", "content": user_request},
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,


            )
        except (KeyError, json.JSONDecodeError) as e:
            raise ValueError(f"Invalid response from OpenAI: {e}")
        except OpenAIError as e:
            raise ValueError(f"OpenAI error: {e}")

        # str() to match quotes
        response_text =  str(response.choices[0].message.content).strip()

        if not response_text:
            print(f"LLM answered {response_text} so function cannot be applied.")
            return {}
        return json.loads(response_text)

    def apply_commands(self,
            df: DataFrame,
            commands: list[dict[str, Any]],
    ) -> DataFrame:
        """
        Apply a series of commands to a DataFrame in sequence.

        :param df: The DataFrame to transform.
        :param commands: A list of commands, each with 'command' and 'kwargs' entries.
        :return: The transformed DataFrame.
        :raises ValueError: If a command is missing, unsupported, or fails to apply.
        """
        try:
            for command in commands:
                print(f"Applying command {command.values()}")
                if (command_name := command.get("command")) is None:
                    raise ValueError("No command was found.")
                elif command_name not in self.commands:
                    raise ValueError(f"Given command '{command_name}' is not supported.")

                args = command.get("kwargs") or {}
                df = self.commands[command_name](df, **args)
        except Exception as e:
            raise ValueError(f"Failed to apply command '{command_name}'"
                             f" with Args '{args}': {e},"
                             f"Error message: {e}") from e
        return df

    def execute_commands(self,
            df: DataFrame,
            user_request: str,
            engine: str = "gpt-4",
            max_tokens: int = 1000,
            temperature: float = 0.7,
            top_p: float = 1,
            max_retries: int = 1,
    ) -> DataFrame:
        """
        Generate and apply commands to a DataFrame based on a user request.

        :param df: The DataFrame to transform.
        :param user_request: The user's direct request (in natural language).
        :param engine: The model/engine to use (default 'gpt-4').
        :param max_tokens: The maximum number of tokens in the response.
        :param temperature: The sampling temperature.
        :param top_p: The nucleus sampling parameter.
        :param max_retries: How many times to retry if there's an error.
        :return: The transformed DataFrame after applying commands.
        """
        retry_count = 0
        prompt = self._get_prompt()

        commands = self.generate_commands(
                prompt,
                user_request,
                engine,
                max_tokens,
                temperature,
                top_p
        ).get("commands")

        while True:
            try:
                if not commands:
                    print("No commands were provided.")
                    return df
                df = self.apply_commands(df, commands)
                break
            except Exception as e:
                print(f"Ajajaj neco se pokazilo, smula: {e}")
                if retry_count < max_retries:
                    retry_count += 1
                    prompt = self._get_error_prompt(error_message=str(e))
                    commands = self.generate_commands(
                            prompt, f"fix {commands}", engine, max_tokens, temperature, top_p
                    ).get("commands")
                else:
                    print("Blame the developer here.")
                    raise
        return df