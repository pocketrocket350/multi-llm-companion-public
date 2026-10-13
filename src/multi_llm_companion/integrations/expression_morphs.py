"""意味上の表情プリセットを、モデル固有のPMXモーフへ解決する。"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ResolvedExpressionMorphs:
    """一つの表情プリセットをPMX名と最終強度へ解決した結果。"""

    preset_name: str = ""
    morph_weights: tuple[tuple[str, float], ...] = ()

    @property
    def primary_morph(self) -> str:
        return self.morph_weights[0][0] if self.morph_weights else ""


class ExpressionMorphResolver:
    """キャラクター共通表情とPMX固有名の境界を一か所で扱う。"""

    def __init__(
        self,
        *,
        expression_presets: Mapping[str, Mapping[str, float]] | None = None,
        emotion_presets: Mapping[str, str] | None = None,
        emotion_preset_variants: Mapping[str, tuple[str, ...] | list[str]] | None = None,
        model_morphs: Mapping[str, str] | None = None,
        legacy_emotion_morphs: Mapping[str, str] | None = None,
        fixed_strength_morphs: Iterable[str] | None = None,
        random_source: Callable[[], float] = random.random,
    ) -> None:
        self._expression_presets = {
            name.strip().lower(): dict(morphs)
            for name, morphs in (expression_presets or {}).items()
        }
        self._emotion_presets = {
            emotion.strip().lower(): preset.strip().lower()
            for emotion, preset in (emotion_presets or {}).items()
        }
        self._emotion_preset_variants = {
            emotion.strip().lower(): tuple(
                preset.strip().lower() for preset in presets if preset.strip()
            )
            for emotion, presets in (emotion_preset_variants or {}).items()
            if emotion.strip()
        }
        self._model_morphs = {
            role.strip().lower(): morph.strip()
            for role, morph in (model_morphs or {}).items()
            if role.strip() and morph.strip()
        }
        self._legacy_emotion_morphs = {
            emotion.strip().lower(): morph.strip()
            for emotion, morph in (legacy_emotion_morphs or {}).items()
            if emotion.strip() and morph.strip()
        }
        self._fixed_strength_morphs = {
            morph.strip() for morph in (fixed_strength_morphs or ()) if morph.strip()
        }
        self._random_source = random_source
        self._last_preset_by_emotion: dict[str, str] = {}

    def resolve(
        self,
        emotion: str,
        intensity: float,
        *,
        advance_variant: bool = True,
    ) -> ResolvedExpressionMorphs:
        """感情をプリセット経由で解決し、未設定時だけ旧対応表へ戻す。"""

        emotion_name = emotion.strip().lower() or "neutral"
        safe_intensity = max(0.0, min(1.0, intensity))
        preset_name = self._select_preset(emotion_name, advance_variant=advance_variant)
        preset = self._expression_presets.get(preset_name)
        if preset is not None:
            # neutral_softのような基礎表情は、感情強度0でも設定値を保つ。
            scale = 1.0 if emotion_name == "neutral" else safe_intensity
            combined: dict[str, float] = {}
            for role, base_weight in preset.items():
                morph_name = self._model_morphs.get(role.strip().lower(), "")
                if not morph_name:
                    continue
                normalized_weight = max(0.0, min(1.0, base_weight))
                applied_weight = normalized_weight * scale
                # 隠し位置から表示位置へ移す特殊モーフは、中間値では画面へ現れない。
                if (
                    morph_name in self._fixed_strength_morphs
                    and normalized_weight >= 1.0
                    and scale > 0.0
                ):
                    applied_weight = 1.0
                combined[morph_name] = min(
                    1.0,
                    combined.get(morph_name, 0.0) + applied_weight,
                )
            return ResolvedExpressionMorphs(
                preset_name=preset_name,
                morph_weights=tuple(combined.items()),
            )

        legacy_morph = self._legacy_emotion_morphs.get(emotion_name, "")
        if not legacy_morph or safe_intensity <= 0.0:
            return ResolvedExpressionMorphs()
        return ResolvedExpressionMorphs(
            morph_weights=((legacy_morph, safe_intensity),),
        )

    def reset(self) -> None:
        """会話終了時に直近表情の履歴を破棄する。"""

        self._last_preset_by_emotion.clear()

    def _select_preset(self, emotion_name: str, *, advance_variant: bool) -> str:
        variants = self._emotion_preset_variants.get(emotion_name)
        if not variants:
            return self._emotion_presets.get(emotion_name, "")
        if not advance_variant:
            return self._last_preset_by_emotion.get(emotion_name, variants[0])
        eligible = tuple(
            preset
            for preset in variants
            if preset != self._last_preset_by_emotion.get(emotion_name)
        )
        candidates = eligible or variants
        ratio = min(max(float(self._random_source()), 0.0), 0.999999999999)
        preset_name = candidates[int(ratio * len(candidates))]
        self._last_preset_by_emotion[emotion_name] = preset_name
        return preset_name
