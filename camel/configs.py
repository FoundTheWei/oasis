from __future__ import annotations

from abc import ABC
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Union

if TYPE_CHECKING:
    from camel.functions import OpenAIFunction


@dataclass(frozen=True)
class BaseConfig(ABC):
    pass


@dataclass(frozen=True)
class ChatGPTConfig(BaseConfig):
    r"""Defines the parameters for generating chat completions using the
    OpenAI API.

    Args:
        temperature (float, optional): Sampling temperature to use, between
            :obj:`0` and :obj:`2`. Higher values make the output more random,
            while lower values make it more focused and deterministic.
            (default: :obj:`0.2`)
        top_p (float, optional): An alternative to sampling with temperature,
            called nucleus sampling, where the model considers the results of
            the tokens with top_p probability mass. So :obj:`0.1` means only
            the tokens comprising the top 10% probability mass are considered.
            (default: :obj:`1.0`)
        n (int, optional): How many chat completion choices to generate for
            each input message. (default: :obj:`1`)
        stream (bool, optional): If True, partial message deltas will be sent
            as data-only server-sent events as they become available.
            (default: :obj:`False`)
        stop (str or list, optional): Up to :obj:`4` sequences where the API
            will stop generating further tokens. (default: :obj:`None`)
        max_tokens (int, optional): The maximum number of tokens to generate
            in the chat completion. The total length of input tokens and
            generated tokens is limited by the model's context length.
            (default: :obj:`None`)
        presence_penalty (float, optional): Number between :obj:`-2.0` and
            :obj:`2.0`. Positive values penalize new tokens based on whether
            they appear in the text so far, increasing the model's likelihood
            to talk about new topics. See more information about frequency and
            presence penalties. (default: :obj:`0.0`)
        frequency_penalty (float, optional): Number between :obj:`-2.0` and
            :obj:`2.0`. Positive values penalize new tokens based on their
            existing frequency in the text so far, decreasing the model's
            likelihood to repeat the same line verbatim. See more information
            about frequency and presence penalties. (default: :obj:`0.0`)
        logit_bias (dict, optional): Modify the likelihood of specified tokens
            appearing in the completion. Accepts a json object that maps tokens
            (specified by their token ID in the tokenizer) to an associated
            bias value from :obj:`-100` to :obj:`100`. Mathematically, the bias
            is added to the logits generated by the model prior to sampling.
            The exact effect will vary per model, but values between:obj:` -1`
            and :obj:`1` should decrease or increase likelihood of selection;
            values like :obj:`-100` or :obj:`100` should result in a ban or
            exclusive selection of the relevant token. (default: :obj:`{}`)
        user (str, optional): A unique identifier representing your end-user,
            which can help OpenAI to monitor and detect abuse.
            (default: :obj:`""`)
    """
    temperature: float = 0.2  # openai default: 1.0
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    stop: Optional[Union[str, Sequence[str]]] = None
    max_tokens: Optional[int] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    logit_bias: Dict = field(default_factory=dict)
    user: str = ""


@dataclass(frozen=True)
class FunctionCallingConfig(ChatGPTConfig):
    r"""Defines the parameters for generating chat completions using the
    OpenAI API with functions included.

    Args:
        functions (List[Dict[str, Any]]): A list of functions the model may
            generate JSON inputs for.
        function_call (Union[Dict[str, str], str], optional): Controls how the
            model responds to function calls. :obj:`"none"` means the model
            does not call a function, and responds to the end-user.
            :obj:`"auto"` means the model can pick between an end-user or
            calling a function. Specifying a particular function via
            :obj:`{"name": "my_function"}` forces the model to call that
            function. (default: :obj:`"auto"`)
    """
    functions: List[Dict[str, Any]] = field(default_factory=list)
    function_call: Union[Dict[str, str], str] = "auto"

    @classmethod
    def from_openai_function_list(
        cls,
        function_list: List[OpenAIFunction],
        function_call: Union[Dict[str, str], str] = "auto",
        kwargs: Optional[Dict[str, Any]] = None,
    ):
        r"""Class method for creating an instance given the function-related
        arguments.

        Args:
            function_list (List[OpenAIFunction]): The list of function objects
                to be loaded into this configuration and passed to the model.
            function_call (Union[Dict[str, str], str], optional): Controls how
                the model responds to function calls, as specified in the
                creator's documentation.
            kwargs (Optional[Dict[str, Any]]): The extra modifications to be
                made on the original settings defined in :obj:`ChatGPTConfig`.

        Return:
            FunctionCallingConfig: A new instance which loads the given
                function list into a list of dictionaries and the input
                :obj:`function_call` argument.
        """
        return cls(
            functions=[
                func.get_openai_function_schema() for func in function_list
            ],
            function_call=function_call,
            **(kwargs or {}),
        )


@dataclass(frozen=True)
class OpenSourceConfig(BaseConfig):
    r"""Defines parameters for setting up open-source models and includes
    parameters to be passed to chat completion function of OpenAI API.

    Args:
        model_path (str): The path to a local folder containing the model
            files or the model card in HuggingFace hub.
        server_url (str): The URL to the server running the model inference
            which will be used as the API base of OpenAI API.
        api_params (ChatGPTConfig): An instance of :obj:ChatGPTConfig to
            contain the arguments to be passed to OpenAI API.
    """
    model_path: str
    server_url: str
    api_params: ChatGPTConfig = ChatGPTConfig()


OPENAI_API_PARAMS = {param for param in asdict(ChatGPTConfig()).keys()}
OPENAI_API_PARAMS_WITH_FUNCTIONS = {
    param
    for param in asdict(FunctionCallingConfig()).keys()
}