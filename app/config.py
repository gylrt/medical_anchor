from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    chroma_dir: str = "./data/chroma"
    collection_name: str = "medlineplus_v1"
    dailymed_collection_name: str = "dailymed_v1"
    dailymed_name_index_path: str = "./data/processed/dailymed_name_index.json"
    embed_model: str = "BAAI/bge-small-en-v1.5"
    ner_model: str = "samrawal/bert-base-uncased_clinical-ner"
    ner_min_score: float = 0.95
    max_distance: float = 0.25
    mesh_match_threshold: float = 85.0
    batch_size: int = 256
    max_text_length: int = 10_000

    model_config = {"env_file": ".env"}


settings = Settings()
