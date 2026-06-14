"""Typed configuration loader.

Loads settings from (in priority order):
  1. Init arguments        (highest — used in tests)
  2. Environment variables (e.g. PARSING__DEFAULT_STRATEGY=hi_res)
  3. .env file at project root
  4. config/default.yaml   (lowest — base config)
"""
from functools import lru_cache
from typing import Tuple, Type

from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

from src.common.paths import CONFIG_DIR, PROJECT_ROOT


class ParsingConfig(BaseModel):
    default_strategy: str = "auto"
    ocr_languages: list[str] = ["eng"]
    pdf_infer_table_structure: bool = True
    extract_image_block_types: list[str] = ["Image", "Table"]


class ChunkingConfig(BaseModel):
    target_size_tokens: int = 400
    max_size_tokens: int = 700
    min_size_tokens: int = 100
    overlap_tokens: int = 50
    tokenizer_encoding: str = "cl100k_base"
    drop_element_types: list[str] = ["Footer", "Header", "PageNumber", "PageBreak"]
    min_element_chars: int = 5
    isolated_element_types: list[str] = ["Table", "Image"]


class EmbeddingConfig(BaseModel):
    model: str = "text-embedding-3-small"
    dimensions: int = 1536
    batch_size: int = 100
    # Prepend the chunk's section_path breadcrumb to the embedded text so the
    # vector reflects where the content lives, not just the content itself.
    include_section_path: bool = True
    # Passed to the OpenAI client; it retries transient and rate-limit errors.
    max_retries: int = 5


class StorageConfig(BaseModel):
    provider: str = "chroma"
    # Name of the Chroma collection that holds this corpus's vectors.
    collection_name: str = "rag_chunks"
    # Distance metric for nearest-neighbour search. cosine matches the
    # normalized vectors text-embedding-3-* returns.
    distance: str = "cosine"


class RetrievalConfig(BaseModel):
    # How many chunks a search returns by default.
    top_k: int = 5
    # Drop hits whose cosine similarity is below this. 0.0 keeps everything
    # with non-negative similarity; raise it to filter out weak matches.
    min_score: float = 0.0


class RerankConfig(BaseModel):
    # Whether the pipeline reranks retrieval results before using them.
    enabled: bool = True
    # Chat model that scores each (question, chunk) pair for relevance.
    model: str = "gpt-4o-mini"
    # How many candidates to pull from retrieval and feed into reranking
    # (a wide net, so the right chunk is in the pile even if ranked low).
    candidate_k: int = 20
    # How many chunks to keep after reranking re-sorts them.
    top_n: int = 5
    # How many candidates to score concurrently. Reranking makes one model call
    # per candidate; running them in parallel cuts request latency sharply.
    max_workers: int = 10
    # Passed to the OpenAI client; it retries transient and rate-limit errors.
    max_retries: int = 5


class GenerationConfig(BaseModel):
    # Chat model that writes the final answer from the retrieved context.
    model: str = "gpt-4o"
    # 0.0 keeps answers deterministic and grounded; raise for more freedom.
    temperature: float = 0.0
    # Upper bound on answer length, in tokens.
    max_tokens: int = 700
    # Passed to the OpenAI client; it retries transient and rate-limit errors.
    max_retries: int = 5


class EvaluationConfig(BaseModel):
    # Chat model used as the judge that scores answer correctness/faithfulness.
    judge_model: str = "gpt-4o"
    # Rank cutoff for retrieval metrics (hit-rate@k, MRR considers up to here).
    k: int = 5
    # Passed to the OpenAI client; it retries transient and rate-limit errors.
    max_retries: int = 5


class LoggingConfig(BaseModel):
    level: str = "INFO"
    to_file: bool = True
    file_rotation: str = "00:00"
    file_retention: str = "14 days"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        yaml_file=str(CONFIG_DIR / "default.yaml"),
        extra="ignore",
        case_sensitive=False,
    )

    openai_api_key: str = Field(..., description="OpenAI API key for LLM calls")

    parsing: ParsingConfig = Field(default_factory=ParsingConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    rerank: RerankConfig = Field(default_factory=RerankConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls),
            file_secret_settings,
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
