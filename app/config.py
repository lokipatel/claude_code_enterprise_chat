from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    neo4j_uri: str
    neo4j_username: str
    neo4j_password: str
    neo4j_database: str


settings = Settings()
