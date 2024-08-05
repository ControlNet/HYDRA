from typing import List, Literal
from dataclasses import dataclass


@dataclass
class ExecutionResult:
    type: Literal["continue", "final", "error"]
    feedbacks: List[str]
    variables: List[str]  # name: description
    variable_names: List[str]  # name only
    final_result: str

    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            type=data["type"],
            feedbacks=data["feedbacks"],
            variables=data["variables"],
            variable_names=data["variable_names"],
            final_result=data["final_result"],
        )

    def to_dict(self):
        return {
            "type": self.type,
            "feedbacks": self.feedbacks,
            "variables": self.variables,
            "variable_names": self.variable_names,
            "final_result": self.final_result,
        }


@dataclass
class ExecutionRequest:
    code: str
    send_image: bool

    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            code=data["code"],
            send_image=data["send_image"],
        )

    def to_dict(self):
        return {
            "code": self.code,
            "send_image": self.send_image,
        }
