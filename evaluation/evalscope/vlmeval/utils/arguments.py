from dataclasses import dataclass, field
from typing import Optional, List, Union
from os import environ
from vlmeval.smp import *

logger = get_logger("arguments")


@dataclass
class Arguments:
    data: List[str] = field(default_factory=list)
    model: Union[List[dict], List[str]] = field(default_factory=dict)
    fps: int = -1
    nframe: int = 8
    pack: bool = False
    use_subtitle: bool = False
    work_dir: str = "outputs"
    mode: str = "all"
    nproc: int = 16
    retry: Optional[int] = None
    judge: Optional[str] = None
    verbose: bool = False
    ignore: bool = False
    reuse: bool = False
    limit: Optional[int] = None
    config: Optional[str] = None
    judge_args: Optional[str] = None

    # For OpenAI API
    OPENAI_API_KEY: str = "EMPTY"
    OPENAI_API_BASE: Optional[str] = None
    LOCAL_LLM: Optional[str] = None

    def __post_init__(self):
        try:
            if self.OPENAI_API_BASE and self.LOCAL_LLM:
                environ.update(
                    {
                        "OPENAI_API_KEY": self.OPENAI_API_KEY,
                        "OPENAI_API_BASE": self.OPENAI_API_BASE,
                        "LOCAL_LLM": self.LOCAL_LLM,
                    }
                )
        except Exception as e:
            logger.error(f"Error occurred when setting environment variables: {e}")
            raise e
