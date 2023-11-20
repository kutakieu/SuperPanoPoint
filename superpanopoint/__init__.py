import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    環境変数を読み取り保持するクラス
    アルファベットの大文字小文字問わず、メンバー変数名と同じ環境変数名が存在する場合はその値が読み取られる
    """

    config_dir: str = Field(default="config")
    config_name: str = Field(default="config.yaml")

    data_dir: Path = Field(default="data")
    img_dir_name: str = Field(default="imgs")
    points_dir_name: str = Field(default="points")

    img_width_key: str = Field(default="img_w")
    img_height_key: str = Field(default="img_h")
    points_key: str = Field(default="points")

    # 主にローカルでの開発時に.envファイルを参照するように設定
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
