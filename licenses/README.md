# 同梱する物理DLLのライセンス

対象：`src/multi_llm_companion/ui/native/UMTNativePlugin.dll`。SHA256：`a69dbc3e55dd276f49bf07a330843bc8cdfc1ac78a1845c728858a6f7db777fe`。

| 部分 | 作者・出典 | 同梱する本文 | 改変・条件 |
| --- | --- | --- | --- |
| UMTNativePlugin | Candidum Games、[対象コミット](https://github.com/CandidumGames/UMTNativePlugin/tree/cf0d4c857044e29ba9ee90c915577707ecbeae3a) | `UMTNativePlugin-MIT.txt`。同コミットのLICENSEをそのまま保存 | MIT。著作権・許諾本文を保持する |
| Bullet 2.75 | Erwin Coumans、[上流が指定する保存アーカイブ](https://storage.googleapis.com/google-code-archive-downloads/v2/code.google.com/bullet/bullet-2.75.zip) | `Bullet-2.75-Zlib.txt`。`bullet-2.75/LICENSE`をそのまま保存 | zlib。出所を偽らず、変更を明記し、通知を除去しない |

BulletアーカイブのSHA256は、対象UMTコミットのCMake定義と一致する`fe2e369593b034c00a1245676edd8417a58c8030a94e061220f0c228ca61fe8f`。UnityMMDToolsの[Bullet 2.75通知](https://github.com/CandidumGames/UnityMMDTools/blob/d5680009a5940f1042ecb5a82ccd696dbfd0cd46/Third%20Party%20Notices.md#bullet-physics-sdk)とも利用版と著作権を照合した。

既存DLLは、対象UMTコミットとBullet 2.75をローカルビルドし、Bulletの追加線形減衰しきい値に `tools/umt_native_unit_scaled_damping.patch` の2行変更を適用した記録を持つ。元ソフトの著作者は上記の作者であり、本プロジェクトの変更を上流の原本と扱わない。パッチも候補へ同梱する。既存DLLの出所・ビルド記録は `THIRD_PARTY_NOTICES.md` に保持する。

このソース候補にPython、Qt/PySide6、Panda3D等のパッケージ本体は含めない。セットアップ時に取得するパッケージにはそれぞれのライセンスが適用される。将来バイナリ一式を再配布する場合は、実際に同梱する版の本文・NOTICEと必要条件を別途確認する。
