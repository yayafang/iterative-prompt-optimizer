from typing import List, Literal, Union
import dataclasses
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

MessageRole = Literal["system", "user", "assistant"]


@dataclasses.dataclass()
class Message:
    role: MessageRole
    content: str

    def to_str(self) -> str:
        return f"{self.role}: {self.content}"

    @staticmethod
    def messages_to_str(messages: List["Message"]) -> str:
        return "\n".join([message.to_str(message) for message in messages])


class ModelBase:
    def __init__(self, name: str):
        self.name = name
        self.client = OpenAI()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"

    @retry(wait=wait_random_exponential(min=1, max=180), stop=stop_after_attempt(6))
    def generate_chat(
        self,
        messages: List[Message],
        max_tokens: int = 1024,
        temperature: float = 0.0,
        num_comps: int = 1,
        seed: int = 42,
    ) -> Union[List[str], str]:
        response = self.client.chat.completions.create(
            model=self.name,
            messages=[dataclasses.asdict(message) for message in messages],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=1,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            n=num_comps,
            seed=seed,
        )
        if num_comps == 1:
            return response.choices[0].message.content
        return [choice.message.content for choice in response.choices]


def get_model(model_name: str) -> ModelBase:
    """Get the appropriate model instance based on model name."""
    if model_name == "gpt4-turbo":
        return GPT4()
    elif model_name == "gpt-4o":
        return GPT4O()
    elif model_name == "gpt-35-turbo":
        return GPT35()
    else:
        raise ValueError(f"Unknown model: {model_name}")


class GPT4(ModelBase):
    def __init__(self):
        super().__init__("gpt-4-1106-preview")


class GPT4O(ModelBase):
    def __init__(self):
        super().__init__("gpt-4o-2024-08-06")


class GPT35(ModelBase):
    def __init__(self):
        super().__init__("gpt-35-turbo")
