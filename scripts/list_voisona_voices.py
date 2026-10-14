"""合成せずに、導入済みボイスのAPI上の名前だけを確認する。"""

from __future__ import annotations

import json
import os
from pathlib import Path

import httpx
import yaml

from multi_llm_companion.config import AudioConfig
from multi_llm_companion.integrations.voisona_talk import VoiSonaTalkVoiceOutput


def main(config_path: Path | None = None) -> int:
    path = config_path or Path(__file__).resolve().parents[1] / "local/audio.yaml"
    try:
        config = AudioConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
        settings = config.voice_output
        username = os.environ.get(settings.username_env, "")
        password = os.environ.get(settings.password_env, "")
        if not username or not password:
            print("API用ユーザー名・パスワードの環境変数を設定し、ターミナルを開き直してください。")
            return 1
        with httpx.Client(
            auth=httpx.BasicAuth(username, password), timeout=settings.timeout
        ) as client:
            response = client.get(f"{settings.base_url.rstrip('/')}/voices")
            response.raise_for_status()
            voices = VoiSonaTalkVoiceOutput._voice_items(response.json())
        names = list(dict.fromkeys(str(item.get("voice_name", "")).strip() for item in voices))
        names = [name for name in names if name]
        if not names:
            print("利用できるボイスがありません。VoiSona Talkでボイスの導入を確認してください。")
            return 1
        print("利用できる voice_name（引用符の内側を環境変数へ設定）:")
        for name in names:
            print(json.dumps(name, ensure_ascii=False))
        print("合成・再生・設定保存は行っていません。使用するボイスの名前を選んでください。")
        return 0
    except (OSError, ValueError, yaml.YAMLError, httpx.HTTPError):
        print(
            "ボイス一覧を取得できません。local/audio.yaml、起動状態、REST API設定を確認してください。"
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
