# -*- coding: utf-8 -*-
# pylint: disable=C0301
"""An agent class that implements the CodeAct agent.
This agent can execute code interactively as actions.
More details can be found at the paper of CodeAct agent
https://arxiv.org/abs/2402.01030
and the original repo of codeact https://github.com/xingyaoww/code-act
"""
import base64
import pickle as pkl

from agentscope.agents import AgentBase
from agentscope.message import Msg
from agentscope.service import (
    ServiceResponse,
    ServiceExecStatus,
    NoteBookExecutor,
)
from agentscope.parsers import RegexTaggedContentParser

SYSTEM_MESSAGE = """
You are a helpful assistant that gives helpful, detailed, and polite answers to the user's questions.
The code written by assistant should be enclosed using <execute> tag, for example: <execute> print('Hello World!') </execute>.
You should provide the solution in a single <execute> block instead of taking many turns. You'll recieve feedback from your code execution. You should always import packages and define variables before starting to use them.
You should stop <execute> and provide an answer when they have already obtained the answer from the execution result. Whenever possible, execute the code for the user using <execute> instead of providing it.
Your response should be concise, but do express their thoughts. Always write the code in <execute> block to execute them.
You should not ask for the user's input unless necessary. Solve the task on your own and leave no unanswered questions behind.
You should do every thing by your self. You are not allowed to install any new packages or overwrite abailable variables provided to you in the question.
"""  # noqa

EXAMPLE_MESSAGE = """
Additionally, you are provided with the following variables available:
{example_code}
The above variables is already available in your interactive Python (Jupyter Notebook) environment, allowing you to directly use them without needing to redeclare them.
"""  # noqa


class MyNotebookExecutor(NoteBookExecutor):

    def inject_available_variables(self, variables: dict) -> ServiceResponse:
        """
        Inject available variables to the notebook.
        """
        serialized_data = base64.b64encode(pkl.dumps(variables)).decode('utf-8')
        code = f"""
    # Deserialize and inject variables
    import pickle
    import base64
    serialized_data = '{serialized_data}'
    var_dict = pickle.loads(base64.b64decode(serialized_data))
    globals().update(var_dict)
    """
        return self.run_code_on_notebook(code)

    def access_variable(self, variable_name: str) -> ServiceResponse:
        """
        Access variable from the notebook.
        """
        code = f"""
    import pickle
    import base64
    variable_data = pickle.dumps(globals().get('{variable_name}', 'Variable not found'))
    encoded_data = base64.b64encode(variable_data).decode('utf-8')
    print(encoded_data)
    """
        result = self.run_code_on_notebook(code)
        if result.status == ServiceExecStatus.SUCCESS:
            variable = pkl.loads(base64.b64decode(result.content[0]))
            return variable
        else:
            return None


class CodeActAgent(AgentBase):
    """
    The implementation of CodeAct-agent.
    The agent can execute code interactively as actions.
    More details can be found at the paper of codeact agent
    https://arxiv.org/abs/2402.01030
    and the original repo of codeact https://github.com/xingyaoww/code-act
    """

    def __init__(
        self,
        name: str,
        model_config_name: str,
        example_code: str = "",
    ) -> None:
        """
        Initialize the CodeActAgent.
        Args:
            name(`str`):
                The name of the agent.
            model_config_name(`str`):
                The name of the model configuration.
            example_code(Optional`str`):
                The example code to be executed before the interaction.
                You can import reference libs, define variables
                and functions to be called. For example:

                    ```python
                    from agentscope.service import bing_search
                    import os

                    api_key = "{YOUR_BING_API_KEY}"

                    def search(question: str):
                        return bing_search(question, api_key=api_key, num_results=3).content
                    ```

        """  # noqa
        super().__init__(
            name=name,
            model_config_name=model_config_name,
        )
        self.n_max_executions = 5
        self.example_code = example_code
        self.code_executor = MyNotebookExecutor()

        sys_msg = Msg(name="user", role="user", content=SYSTEM_MESSAGE)
        example_msg = Msg(
            name="user",
            role="user",
            content=EXAMPLE_MESSAGE.format(example_code=self.example_code),
        )

        self.memory.add(sys_msg)
        self.memory.add(example_msg)
        # print(sys_msg.content, example_msg.content) 

        # if self.example_code != "":
        #     code_execution_result = self.code_executor.run_code_on_notebook(
        #         self.example_code,
        #     )
        #     code_exec_msg = self.handle_code_result(
        #         code_execution_result,
        #         "Example Code executed: ",
        #     )
        #     self.memory.add(example_msg)
        #     self.memory.add(code_exec_msg)
        #     self.speak(code_exec_msg)

        self.parser = RegexTaggedContentParser(try_parse_json=False)

    def handle_code_result(
        self,
        code_execution_result: ServiceResponse,
        content_pre_sring: str = "",
    ) -> Msg:
        """return the message from code result"""
        code_exec_content = content_pre_sring
        if code_execution_result.status == ServiceExecStatus.SUCCESS:
            code_exec_content += "Execution Successful:\n"
        else:
            code_exec_content += "Execution Failed:\n"
        code_exec_content += "Execution Output:\n" + str(
            code_execution_result.content,
        )
        if len(code_exec_content)> 4096:
            code_exec_content = code_exec_content[:2048] + "... [truncated due to length, do not print too much output as it exceeds your context length]"
        if code_execution_result.status != ServiceExecStatus.SUCCESS:
            code_exec_content +="\nSide note: Remember to enclose your code in <execute> </execute> tag and do not overwrite any available variables provided to you in the question, especially that they contain the data."
        return Msg(name="user", role="user", content=code_exec_content)

    def reply(self, x: Msg = None) -> Msg:
        """The reply function that implements the codeact agent."""

        self.memory.add(x)

        execution_count = 0
        while (
            self.memory.get_memory(1)[-1].role == "user"
            and execution_count < self.n_max_executions
        ):
            prompt = self.model.format(self.memory.get_memory())
            model_res = self.model(prompt)
            msg_res = Msg(
                name=self.name,
                content=model_res.text,
                role="assistant",
            )
            self.memory.add(msg_res)
            self.speak(msg_res)
            res = self.parser.parse(model_res)
            code = res.parsed.get("execute")
            if code is not None:
                code = code.strip()
                code_execution_result = (
                    self.code_executor.run_code_on_notebook(code)
                )
                execution_count += 1
                code_exec_msg = self.handle_code_result(code_execution_result)
                self.memory.add(code_exec_msg)
                self.speak(code_exec_msg)

        if execution_count == self.n_max_executions:
            assert self.memory.get_memory(1)[-1].role == "user"
            code_max_exec_msg = Msg(
                name="assistant",
                role="assistant",
                content=(
                    "I have reached the maximum number "
                    f"of executions ({self.n_max_executions=}). "
                    "Can you assist me or ask me another question?"
                ),
            )
            self.memory.add(code_max_exec_msg)
            self.speak(code_max_exec_msg)
            return code_max_exec_msg

        return msg_res
