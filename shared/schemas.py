from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class TopicSpec:
    id: str
    question: str
    audience: str
    constraints: Dict[str, Any]


@dataclass
class OutputConfig:
    include_short_form: bool
    short_form_platforms: List[str]


@dataclass
class RunConfig:
    run_id: str
    topics: List[TopicSpec]
    output: OutputConfig
