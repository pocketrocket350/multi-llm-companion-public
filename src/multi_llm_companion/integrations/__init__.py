"""第2段階以降の外部機能を疎結合に保つ入口。"""

from .base import (
    AudioInputDevice,
    AvatarControlPort,
    AvatarModelPreview,
    LipSyncState,
    MmdAvatarCommand,
    MmdRendererPort,
    MmdRendererStatus,
    SpeechRecognitionPort,
    VoiceOutputPort,
)
from .factory import build_avatar_controller, build_speech_recognizer, build_voice_output
from .gesture_selection import GestureCandidate, GestureSelection, GestureSelector
from .mmd_renderer import (
    LegacyPmxPreviewRenderer,
    MmdRuntimeRenderer,
    RendererBackedAvatarController,
)
from .mock_avatar import Mock3DAvatarController
from .noop import NullAvatarController, NullSpeechRecognizer, NullVoiceOutput
from .pmx_preview import PmxPreviewAvatarController, PmxPreviewError, read_pmx_preview
from .vmd_catalog import (
    ResolvedVmdCatalogMotion,
    VmdCatalogError,
    inspect_vmd_motion_library,
    list_vmd_catalog_actions,
    load_vmd_catalog_motion,
    load_vmd_catalog_motions,
    load_vmd_motion_library,
    save_vmd_catalog_motion_enabled,
    validate_vmd_model_compatibility,
)
from .voisona_talk import VoiSonaTalkVoiceOutput
from .vosk_recognizer import VoskSpeechRecognizer
from .whisper_recognizer import WhisperSpeechRecognizer

__all__ = [
    "AvatarControlPort",
    "AvatarModelPreview",
    "LipSyncState",
    "AudioInputDevice",
    "MmdAvatarCommand",
    "MmdRendererPort",
    "MmdRendererStatus",
    "VoiSonaTalkVoiceOutput",
    "VoskSpeechRecognizer",
    "WhisperSpeechRecognizer",
    "Mock3DAvatarController",
    "LegacyPmxPreviewRenderer",
    "MmdRuntimeRenderer",
    "RendererBackedAvatarController",
    "PmxPreviewAvatarController",
    "PmxPreviewError",
    "ResolvedVmdCatalogMotion",
    "VmdCatalogError",
    "inspect_vmd_motion_library",
    "build_avatar_controller",
    "build_speech_recognizer",
    "build_voice_output",
    "GestureCandidate",
    "GestureSelection",
    "GestureSelector",
    "read_pmx_preview",
    "load_vmd_catalog_motion",
    "load_vmd_catalog_motions",
    "load_vmd_motion_library",
    "save_vmd_catalog_motion_enabled",
    "list_vmd_catalog_actions",
    "validate_vmd_model_compatibility",
    "NullAvatarController",
    "NullSpeechRecognizer",
    "NullVoiceOutput",
    "SpeechRecognitionPort",
    "VoiceOutputPort",
]
