# Centralized typed settings for the project.
# This version supports:
# 1) OPENAI_API_KEY from environment or .env
# 2) fallback to an external key file, matching the notebook approach

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _get_project_root() -> Path:
    if "__file__" in globals():
        return Path(__file__).resolve().parents[1]
    return Path.cwd().resolve().parents[0]


def _read_openai_key_from_file(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None

    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return None

    # Support either:
    # sk-...
    # or
    # OPENAI_API_KEY=sk-...
    if "=" in raw and raw.startswith("OPENAI_API_KEY"):
        raw = raw.split("=", 1)[1].strip().strip('"').strip("'")

    return raw or None


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    project_root: Path = _get_project_root()

    data_dir: Path = project_root / "data"
    artifacts_dir: Path = data_dir / "artifacts"
    processed_dir: Path = data_dir / "processed"
    eval_dir: Path = data_dir / "eval"
    logs_dir: Path = project_root / "logs"

    documents_path: Path = artifacts_dir / "documents.json"
    embedding_matrix_path: Path = artifacts_dir / "embedding_matrix.npy"
    lexical_index_path: Path = artifacts_dir / "lexical_index.pkl"
    manifest_path: Path = artifacts_dir / "manifest.json"
    all_chunks_path: Path = processed_dir / "all_chunks.json"

    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_key_file: Path = Path(r"C:\projects\OPEN_AI.txt")
    openai_model: str = "gpt-4o-mini"

    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    default_retrieval_strategy: str = "bm25"
    default_top_k: int = 5
    rrf_k: int = 60

    app_name: str = "Course RAG Agent"
    debug: bool = False
    log_level: str = "INFO"

    @property
    def resolved_openai_api_key(self) -> str | None:
        if self.openai_api_key:
            return self.openai_api_key
        return _read_openai_key_from_file(self.openai_key_file)


settings = Settings()