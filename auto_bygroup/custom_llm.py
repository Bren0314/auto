import json
import time
from typing import Any, List, Mapping, Optional, Dict, Union, Tuple
import re
import logging
import requests
from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.utils import get_from_dict_or_env
from pydantic import Field, root_validator
import zhipuai
import openai

openai.api_base = "111"
openai.api_key = "none"
API_KEY = "111"

logger = logging.getLogger(__name__)

def glm3_model(content):
    result_list = list()
    for chunk in openai.ChatCompletion.create(model="chatglm3-6b", messages=content,
                                              stream=True, temperature=0.3, history=[]):
        if hasattr(chunk.choices[0].delta, "content"):
            if chunk.choices[0].delta.content:
                result_list.append(chunk.choices[0].delta.content)
    result = "".join(result_list).strip()

    return result

def zhipu_turbo(content):
    zhipuai.api_key = API_KEY
    response = zhipuai.model_api.sse_invoke(
                                                model="chatglm_turbo",
                                                prompt= content,
                                                temperature=0.3,
                                                top_p=0.7,
                                                incremental=False
                                            )
    *_, last = response.events()
    result = last.data

    return result

class ContextLoader(BaseLoader):
    """
        Process text string.
    """
    def __init__(
        self,
        file_path: str,
        context: str,
    ):
        self.file_path = file_path
        self.context = context

    def load(self) -> List[Document]:
        text = self.context
        metadata = {"source": self.file_path}
        return [Document(page_content=text, metadata=metadata)]
    

class ZhiPuLLm(LLM): 
    model_name: str = Field(default="chatglm_turbo", alias="model")
    request_timeout: Optional[Union[float, Tuple[float, float]]] = None
    temperature: float = 0.1
    top_p: float = 0.7
    api_key: Optional[str] = API_KEY
    query_answer: Optional[str] = None
    streaming: bool = False
    cache: bool = False
    incremental: bool = False
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    online: bool = True

    @property
    def _default_params(self) -> Dict[str, Any]:
        normal_params = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "request_timeout": self.request_timeout,
        }
        return {**normal_params, **self.model_kwargs}

    def _construct_query(self, prompt: str) -> Dict:
        message =  [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
        return message

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {**{"model_name": self.model_name}, **self._default_params}

    @property
    def _llm_type(self) -> str:
        return "chatglm_turbo"

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> str:
        zhipuai.api_key = self.api_key
        if self.query_answer:
            prompt = re.sub(r"Question:.*?\n", "Question: "+self.query_answer+"\n", prompt)
        message = self._construct_query(prompt=prompt)
        
        if self.online:
            ans = zhipu_turbo(message)
        else:
            ans = glm3_model(message)
        if re.search("Observation:.+$", ans, re.S):
            ans = re.search("Observation:.+$", ans, re.S).group().strip()
        return ans

if __name__ == "__main__":
    llm = ZhiPuLLm()
    print(llm("给我讲一个笑话"))
