"""同梱済みモーションを、既存カタログとファイルを上書きせず導入する。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _install_bundled_motions(project_root: Path) -> str:
    local_root = project_root / "local"
    catalog = local_root / "vmd_catalog.yaml"
    if catalog.exists():
        if not catalog.is_file():
            raise ValueError("既存のVMDカタログがファイルではありません。上書きしていません。")
        return "既存のVMDカタログを保持しました。同梱モーションの追加・上書きは行いません。"

    bundled_root = project_root / "bundled"
    manifest = json.loads((bundled_root / "FILE_MANIFEST.json").read_text(encoding="utf-8"))
    entries = manifest["files"]
    paths = [item["path"] for item in entries]
    if len(paths) != len(set(paths)) or "vmd_catalog.yaml" not in paths:
        raise ValueError("同梱モーションの一覧が不正です。")
    pending: list[tuple[Path, bytes]] = []
    for item in entries:
        name = item["path"]
        relative = Path(name)
        permitted = (
            name == "vmd_catalog.yaml"
            or (name.startswith("motions/otokam/") and relative.suffix == ".vmd")
            or (name.startswith("licenses/otokam/") and relative.suffix in {".yaml", ".txt"})
        )
        if not permitted or relative.is_absolute() or ".." in relative.parts:
            raise ValueError("同梱モーションに許可範囲外のパスがあります。")
        source = bundled_root / relative
        target = local_root / relative
        if not source.resolve().is_relative_to(bundled_root.resolve()):
            raise ValueError("同梱モーションの参照先が配布フォルダー外です。")
        if not target.resolve().is_relative_to(local_root.absolute()):
            raise ValueError("同梱モーションの配置先がlocalフォルダー外です。")
        content = source.read_bytes()
        if hashlib.sha256(content).hexdigest() != item["sha256"]:
            raise ValueError(f"同梱モーションのハッシュが一致しません: {name}")
        if target.exists():
            if target.read_bytes() != content:
                raise ValueError(f"既存ファイルと競合しました。上書きしていません: {name}")
        else:
            pending.append((target, content))

    # 全件確認後にコピーし、最後にカタログを有効にする。
    pending.sort(key=lambda item: item[0] == catalog)
    for target, content in pending:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(content)
    return "同梱モーションと出典・規約をlocalへ配置しました。"


if __name__ == "__main__":
    try:
        print(_install_bundled_motions(Path(__file__).resolve().parents[1]))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"[エラー] 同梱モーションを配置できませんでした: {exc}")
        raise SystemExit(1) from exc
