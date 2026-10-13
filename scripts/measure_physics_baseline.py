"""local素材を公開せず、第44段階のPMX物理基準を測定する。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from multi_llm_companion.config import ConfigError, load_runtime_config  # noqa: E402
from multi_llm_companion.integrations.factory import build_avatar_controller  # noqa: E402
from multi_llm_companion.ui.physics_baseline import (  # noqa: E402
    measure_physics_baseline,
    to_physics_ui_preview,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="第44段階のPMX物理性能と安定性を、同じ条件で測定します。"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="匿名化したJSON結果の保存先。省略時は画面だけへ表示します。",
    )
    parser.add_argument(
        "--cadence",
        type=float,
        default=1.0 / 30.0,
        help="重い姿勢更新の周期秒。既定値は現在のアプリと同じ30分の1秒です。",
    )
    args = parser.parse_args()

    try:
        runtime = load_runtime_config(PROJECT_ROOT)
        controller = build_avatar_controller(runtime.avatar, PROJECT_ROOT)
        preview = controller.model_preview
        if preview is None:
            raise RuntimeError("local設定からPMXを読み込めませんでした。3D設定を確認してください。")
        report = measure_physics_baseline(
            to_physics_ui_preview(preview),
            cadence_seconds=args.cadence,
        )
    except (ConfigError, OSError, RuntimeError, ValueError):
        print("[エラー] local設定またはPMX/VMDを読み込めませんでした。詳細は出力しません。")
        return 1

    result = report.as_dict()
    encoded = json.dumps(result, ensure_ascii=False, indent=2)
    print(encoded)
    if args.output is not None:
        output = args.output
        if not output.is_absolute():
            output = PROJECT_ROOT / output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded + "\n", encoding="utf-8")
        print("[保存] 匿名化した測定結果を書き込みました。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
