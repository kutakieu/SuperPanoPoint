from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    config_dir: str = Field(default="config")
    config_name: str = Field(default="config.yaml")

    data_dir: Path = Field(default="data")
    img_dir_name: str = Field(default="imgs")
    points_dir_name: str = Field(default="points")

    img_width_key: str = Field(default="img_w")
    img_height_key: str = Field(default="img_h")
    points_key: str = Field(default="points")

    wandb_api_key: str = Field(default="")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
