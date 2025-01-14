import json
import inspect
from pyexpat.errors import messages
from typing import Any, Callable, Optional

from openai import OpenAI, OpenAIError
from pandas import DataFrame, Series

from src.interface.exceptions import InterfaceException, InterfaceOpenAIException, \
    UnknownModelException
from src.logger import logger


from src.interface.prompts import PROMPT_TEMPLATE, PROMPT_TEMPLATE_ERROR
from src.transforms import trasform_funcs
from private import env

class AIInterface:
    """
    An interface that manages commands, constructs prompts, and communicates
    with OpenAI's API to generate and apply commands on a DataFrame.
    """
    _PROMPT_TEMPLATE: str = PROMPT_TEMPLATE
    _PROMPT_TEMPLATE_ERROR: str = PROMPT_TEMPLATE_ERROR

    _tested_models: list[str] = ["gbt-4"]

    default_settings = {
        "model" : "gpt-4",
        "temperature" : 0.75,
        "top_p" : 0.9,
        "frequancy_penalty" : 0.0,
        "presence_penalty" : 0.0,
    }

    def __init__(self, *,
            load_modules: Optional[list[object]],
            model: Optional[str] = None,
            temperature: Optional[float] = 0.75,
            top_p: Optional[float] = 0.9,
            frequancy_penalty: Optional[float] = 0.0,
            presence_penalty: Optional[float] = 0.0,
    ) -> None:
        logger.debug(f"Initializing {self.__class__.__name__}")
        self._client = OpenAI()
        self.llm_settings(
                model_name=model,
                temperature=temperature,
                top_p=top_p,
                frequancy_penalty=frequancy_penalty,
                presence_penalty=presence_penalty
        )

        self.commands_description: dict[str, str] = {}
        self.commands: dict[str, Callable] = {}

        #primitive cashing

        self._commands_updated = False
        self._available_commands = None


        if load_modules:
            self.add_modules_commands(load_modules)

    def add_command(self, command: Callable) -> None:
        """Adds new command to a class"""
        name = command.__name__
        logger.info(f"Adding command: %s", name)
        signature = inspect.signature(command)
        docstring = inspect.getdoc(command)

        description = f"{signature} - {docstring.replace("\n", "") or "No description"}"
        self._commands_updated = True
        self.commands_description[name] = description
        self.commands[name] = command

    def add_modules_commands(self, modules: list[object]) -> None:
        """parses functions inside a modules in the list and addes them to the class"""
        for module in modules:
            logger.info(f"Adding module: %s", module.__name__)
            for _, func in inspect.getmembers(module, inspect.isfunction):
                self.add_command(func)

    def _get_available_commands(self) -> str:
        """Returns the list of available commands"""
        if self._commands_updated:
            self._available_commands = [
                f"{name}{description}" for name, description in self.commands_description.items()
            ]
        if self._available_commands:
            return "\n".join(self._available_commands)

    def llm_settings(self, *,
            model_name: Optional[str] = None,
            temperature: Optional[float] = None,
            top_p: Optional[float] = None,
            frequancy_penalty: Optional[float] = None,
            presence_penalty: Optional[float] = None,
    ) -> None:
        """Changes the default setting of the class"""
        if model_name:
            if model_name not in self._tested_models:
                raise UnknownModelException(f"Unknown model name {model_name}")
            logger.debug(f"Using model name: {model_name}")
            self.default_settings["model"] = model_name
        if temperature:
            logger.debug(f"Using temperature: {temperature}")
            self.default_settings["temperature"] = temperature
        if top_p:
            logger.debug(f"Using top p: {top_p}")
            self.default_settings["top_p"] = top_p
        if frequancy_penalty:
            logger.debug(f"Using frequancy penalty: {frequancy_penalty}")
            self.default_settings["frequancy_penalty"] = frequancy_penalty
        if presence_penalty:
            logger.debug(f"Using presence penalty: {presence_penalty}")
            self.default_settings["presence_penalty"] = presence_penalty


    def _get_prompt(self):
        """Creating a formatted prompt"""
        return self._PROMPT_TEMPLATE.format(commands=self._get_available_commands())


    @staticmethod
    def _get_messages(
            user_request: str,
            *,
            system: str = None,
            previous: list[dict[str,str]] = None
    ) -> list[dict[str, str|list]]:
        """Create messages request"""

        if system and previous:
            logger.warning("Both 'system' and 'previous' are provided; using 'system' only.")

        result: list[dict[str, str]] = []
        if system:
            logger.debug("System messages requested")
            result.append({
                "role": "system",
                "content" : [{
                    "type" : "text",
                    "text" : system,
                }]
            })
        elif previous:
            logger.debug("Previous messages requested")
            result.extend(previous)
        else:
            raise InterfaceException("Neither 'system' nor 'previous' commands provided to create messages.")

        result.append({
            "role": "user",
            "content" : [{
                "type" : "text",
                "text" : user,
            }]
        })
        return result

    def _send_request(self,
            messages_list: list[dict[str, str]],
            model: str = "gpt-4",
            temperature: float = 1,
            top_p: float = 0.9,
            frequancy_penalty: float = 0.0,
            presence_penalty: float = 0.0,
    ) -> dict[str,dict[str, str]]:
        """Send request to LLM with messages"""
        try:
            response = self._client.chat.completions.create(
                    model=model,
                    messages=messages_list,
                    frequency_penalty=frequancy_penalty,
                    response_format={
                        "type":"text",
                    },
                    temperature=temperature,
                    top_p=top_p,
                    presence_penalty=presence_penalty,
            )
        except KeyError as e:
            raise InterfaceOpenAIException("Invalid keys were provided.") from e
        except OpenAIError as e:
            raise InterfaceOpenAIException("OpenAI API error") from e

        response_text = str(response.choices[0].message.content).strip()

        result = {}
        if not response_text:
            logger.warning("Response did not return anything.")
            #TODO IMPLEMENT LOGIC FOR NO RESPONSE
        try:
            result = json.loads(response_text)
        except json.decoder.JSONDecodeError as e:
            logger.critical("Response is not valid JSON. %s", e)
            #TODO LOGIC FOR HANDLING THAT

        return result


    def _apply_commands(self,
        df: DataFrame,
        commands: dict[str, dict[str, str]] ,
    ) -> DataFrame:
        # backup dataframe
        new_df = df.copy(deep=True)

        for comm_name, comm_args in commands.items():
            logger.info(f"Applying command: %s with %s", comm_name, comm_args)
            if comm_name not in self.commands:
                raise InterfaceException(f"Command {comm_name} not supported.")

            try:
                new_df = self.commands[comm_name](new_df, **comm_args)
            except KeyError as e:
                raise InterfaceException(f"Command {comm_name} with args {comm_args}"
                                         f"gave an error: {e}")
            return new_df

    def transform(self, df: DataFrame, user_request: str) -> DataFrame:
        prompt = self._get_prompt()
        messages_list = self._get_messages(user_request, system=prompt)
        commands = self._send_request(
                messages_list=messages_list,
                model=self.default_settings["model"],
                temperature=self.default_settings["temperature"],
                top_p=self.default_settings["top_p"],
                frequancy_penalty=self.default_settings["presence_penalty"],
        )
        logger.info(f"Commands received: {commands}")
        return self._apply_commands(df, commands)



if __name__ == '__main__':
    user = ("I need to rename the Column x to my Column and also remove the rows with the empty "
            "values and then save this file as test.csv")
    from src.transforms import trasform_funcs as trasform
    x = AIInterface(load_modules=[trasform])
    y = x.transform(df=None, user_request=user)