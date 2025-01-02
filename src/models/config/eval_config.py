import os
from typing import List
from typing import Optional

from pydantic import Field

from src.models.config.base_config import BaseConfig
from src.models.config.train_config import CorpusConfig


class EvalConfig(BaseConfig):
    """
    Configuration for evaluation
    """

    job_type: str = Field(
        default_factory=lambda: os.getenv("JOB_TYPE", "eval"),
        description="Specify the type of run, which is useful when you're grouping runs together."
    )

    group: str = Field(
        default_factory=lambda: os.getenv("GROUP", "evaluation"),
        description="Specify a group to organize individual runs into a larger experiment."
    )

    test_corpora: List[CorpusConfig] = Field(
        ...,
        description="List of test corpora configurations."
    )
    experiment_name: str = Field(
        default="chall",
        description="Name of the experiment."
    )
    experiment_tag: str = Field(
        default="getting_started",
        description="Tag for the experiment to differentiate configurations."
    )
    device: str = Field(
        default="cuda",
        description="Device to run the training on, e.g., 'cuda' or 'cpu'."
    )
    checkpoint: Optional[str] = Field(
        default=None,
        description="Path to a pre-trained model checkpoint to resume training."
    )
    alt_base_path: str = Field(
        default_factory=lambda: os.getenv("ALT_BASE_PATH", "./"),
        description="Alternate base path for data or checkpoints."
    )
    seed: int = Field(
        default=123,
        description="Random seed for reproducibility."
    )
    bf16: bool = Field(
        default=False,
        description="Whether to use bfloat16 for training."
    )
    fp16: bool = Field(
        default=True,
        description="Whether to use float16 for training."
    )
    kenlm_model_name: Optional["str"] = Field(
        default=None,
        description=""
    )
    lm_alpha: float = Field(
        default=0.5,
        description=""
    )
    lm_beta: float = Field(
        default=1.5,
        description=""
    )
    wav2vec_base_model: Optional[str] = Field(
        default=None,
        description="Does not make sense here"
    )



    @property
    def experiment_label(self):
        return f'{self.experiment_name}_{self.experiment_tag}'
