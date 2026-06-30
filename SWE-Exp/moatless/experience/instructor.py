import re
import logging
import json
import json_repair
from typing import Dict, Any
from pydantic import BaseModel, Field, PrivateAttr, model_validator, ValidationError
from moatless.completion.completion import CompletionModel, LLMResponseFormat
from moatless.completion.model import Completion
from tenacity import (
    retry,
    stop_after_attempt,
    wait_fixed,
    retry_if_exception_type,
    before_sleep_log,
)
logger = logging.getLogger(__name__)

class Instructor(BaseModel):
    system_prompt: str = Field(
        ..., description="System prompt to be used for generating completions"
    )
    output_format: str = Field(
        ..., description="Output format to be used for generating completions"
    )
    task: str = Field(
        ..., description="Task for this agent to complete"
    )
    taken_actions: int = Field(
        ..., description="The number of instructions this agent has given to complete the task"
    )
    _completion: CompletionModel = PrivateAttr()
    
    
    def __init__(
        self,
        completion: CompletionModel,
        task: str,
        taken_actions: int,
        system_prompt: str | None = None,
        output_format: str | None = None,
        **data,
    ):
        super().__init__(
            system_prompt=system_prompt,
            output_format=output_format,
            task=task,
            taken_actions=taken_actions,
            **data,
        )
        self._completion = completion


    def simplify(self, content):
        match = re.search(r"```(?:\w+)?\n?(.*?)```", content, re.DOTALL)
        if match:
            return match.group(1).strip()
        else:
            return content

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_fixed(3),
        retry=retry_if_exception_type((KeyError, json.decoder.JSONDecodeError, json.JSONDecodeError, TypeError, RuntimeError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),  # 重试前打印日志
    )
    def _instruct_with_retry(self, input_messages, exp, node_id):
        # messages = self.message_generator.generate(node)
        response = ""
        messages = input_messages.copy()
        if len(messages) == 1 and messages[0].get("content", "").startswith("<task>"):
            messages = []
        messages.insert(0, {"role": "system", "content": self.system_prompt})
        message = f'<task>\n{self.task}\n{exp}\nYou MUST do code modification and finish the task within max {str(self.taken_actions)} actions.\n</task>\n This is the {node_id}-th actions.'
        history_text = "\n".join(str(m.get("content", "")) for m in input_messages)
        if "raise NotImplementedError" in history_text and "Detailed Description" in self.task:
            message += (
                "\n\n<koce_instruction>\n"
                "The target function stub and the full task specification have already been viewed. "
                "Do not request more search or view actions unless a previous modification failed. "
                "The next instruction must be a concrete code modification instruction with type \"modify\".\n"
                "</koce_instruction>"
            )
        messages.append({"role": "user", "content": message})
        messages.append({"role": "user", "content": self.output_format})
        try:
            logger.info(
                "Instructor request node=%s messages=%d chars=%d",
                node_id,
                len(messages),
                sum(len(str(m.get("content", ""))) for m in messages),
            )
            response = self._completion._litellm_base_completion(
                        messages=messages
                    ).choices[0].message.content
            logger.info("Instructor raw response node=%s chars=%d", node_id, len(response or ""))
            response = self.simplify(response)

            def unescape_values(obj):
                if isinstance(obj, dict):
                    return {k: unescape_values(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [unescape_values(v) for v in obj]
                elif isinstance(obj, str) and "\\" in obj:
                    try:
                        return obj.encode().decode("unicode_escape")
                    except UnicodeDecodeError as e:
                        logger.warning(f"Unicode decode error for string: {obj[:100]}... Error: {e}")
                        return obj.replace("\\", " ")
                return obj

            response = json_repair.loads(response)

            cleaned_response = unescape_values(response)
            thoughts, instructions, context, ty = (cleaned_response['thoughts'], cleaned_response['instructions'],
                                                         cleaned_response['context'],
                                                         cleaned_response['type'])
            history_text = "\n".join(str(m.get("content", "")) for m in input_messages)
            if "raise NotImplementedError" in history_text and ty in {"search", "view"}:
                ty = "modify"
                target_match = re.search(r"Target location:\s*`([^`]+)`", self.task)
                target_path = target_match.group(1).split(",")[0] if target_match else "the target file"
                function_match = re.search(r"Target function:\s*`([^`]+)`", self.task)
                function_name = function_match.group(1) if function_match else "the target function"
                instructions = (
                    f"Replace the `{function_name}` stub with a compact valid Python implementation. "
                    "Use the task specification and helper functions already visible in the file. "
                    "Keep the code ASCII only, omit long docstrings, and return only valid Python code in new_str. "
                    f"The old_str is the current `{function_name}` definition with the TODO and raise NotImplementedError."
                )
                context = target_path
        except Exception as e:
            logger.error(f"Error in _instruct_with_retry: {str(e)}")
            logger.error(f"Response: {response}")
            raise

        return thoughts, instructions, context, ty

    def instruct(self, messages, exp, node_id):
        return self._instruct_with_retry(messages, exp, node_id)

    @property
    def completion(self) -> CompletionModel:
        return self._completion

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        dump = super().model_dump(**kwargs)
        dump["completion"] = self._completion.model_dump(**kwargs)
        return dump
