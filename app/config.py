from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    chroma_dir: str = "./data/chroma"
    collection_name: str = "medlineplus_v1"
    embed_model: str = "BAAI/bge-small-en-v1.5"
    ner_model: str = "samrawal/bert-base-uncased_clinical-ner"
    ner_min_score: float = 0.95
    max_distance: float = 0.5
    batch_size: int = 256

    model_config = {"env_file": ".env"}


settings = Settings()