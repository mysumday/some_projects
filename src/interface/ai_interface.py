import json
import inspect
import os
from typing import Optional, Callable, Any

from openai import OpenAI, OpenAIError
from pandas import DataFrame

from src.interface.prompts import PROMPT_TEMPLATE, PROMPT_TEMPLATE_ERROR

class AIInterface:
    PROMPT_TEMPLATE: str = PROMPT_TEMPLATE
    PROMPT_TEMPLATE_ERROR: str = PROMPT_TEMPLATE_ERROR

    def __init__(self, load_modules: Optional[list[object]] = None):
        self._client = OpenAI()
        #COMMANDS
        self.commands_description: dict[str, str] = {}
        self.commands: dict[str, Callable] = {}

        #primitive cashing
        self._reset_available = False
        self._available_commands: str = ""
        if load_modules:
            self.add_commands_from_modules(load_modules)


    def add_command(self,name: str, func: Callable) -> None:
        self._reset_available = True
        print(f"Func found - {name}")
        signature = inspect.signature(func)
        docstring = inspect.getdoc(func)
        description = f"{signature} - {docstring if docstring else 'No description'}"
        self.commands_description[name] = description
        self.commands[name] = func

    def add_commands_from_modules(self, modules: list[object]) -> None:
        for module in modules:
            for name, func in inspect.getmembers(module, inspect.isfunction):
                self.add_command(name, func)

    def _get_available_commands(self) -> str:
        if self._reset_available and self.commands:
            self._available_commands =  "\n".join(
                    [f" - {name}({description.replace("\n","")})" for name, description in
                     self.commands_description.items()]
            )
        return self._available_commands

    def _get_prompt(self) -> str:
        if not self.commands:
            raise ValueError("No commands available")
        return self.PROMPT_TEMPLATE.format(
                commands=self._get_available_commands(),
        )

    def _get_error_prompt(self,
            error_message: str,
    ) -> str:
        return self.PROMPT_TEMPLATE_ERROR.format(
                error_msg=error_message
        )


    def generate_commands(self,
            prompt: str,
            user_request: str,
            engine: str = "o1",
            max_tokens: int = 1000,
            temperature: float = 0.7,
            top_p: float = 1,
    ) -> dict[str, list[dict]]:
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
        if response_text == "Not Possible" or not response_text:
            print(f"LLM answered {response_text} so function cannot be applied.")
            return {}
        return json.loads(response_text)

    def apply_commands(self,
            df: DataFrame,
            commands: list[dict[str, Any]],
    ) -> DataFrame:
        try:
            for command in commands:
                print(f"Applying command {command.values()}")
                if (command_name := command.get("command")) is None:
                    raise ValueError("No command was found.")
                elif command_name not in self.commands:
                    raise ValueError(f"Given command '{command_name}' is not supported.")

                if (args := command.get("kwargs")) is None:
                    args = {}
                df = self.commands[command_name](df, **args)
        except Exception as e:
            raise ValueError(f"Failed to apply command '{command_name or None}'"
                             f" with Args '{args or None}': {e},"
                             f"Error message: {e}")
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
                    prompt = self._get_error_prompt(error_message=e)
                    commands = self.generate_commands(
                            prompt, f"fix {commands}", engine, max_tokens, temperature, top_p
                    ).get("commands")
                else:
                    print("Blame the developer here.")
                    raise
        return df