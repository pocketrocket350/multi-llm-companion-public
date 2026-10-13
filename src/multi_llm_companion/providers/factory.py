"""設定値からプロバイダー実装を選ぶ拡張可能な生成処理。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from .base import ChatProvider
from .errors import ProviderConfigurationError
from .mock import MockProvider
from .openai_compatible import OpenAICompatibleProvider


class ResolvedProviderSettings(Protocol):
    """config.ResolvedProviderConfigとの疎結合な受け渡し形式。"""

    name: str
    provider: str
    base_url: str
    api_key: str
    model: str
    temperature: float
    max_output_tokens: int
    timeout: float


type ProviderBuilder = Callable[[ResolvedProviderSettings], ChatProvider]


class ProviderFactory:
    """将来、会社固有プロバイダーを登録できる生成器。"""

    def __init__(self, *, include_defaults: bool = True) -> None:
        self._builders: dict[str, ProviderBuilder] = {}
        if include_defaults:
            self.register("mock", _build_mock)
            self.register("openai_compatible", _build_openai_compatible)

    def register(self, provider_type: str, builder: ProviderBuilder) -> None:
        """種類名と専用実装の生成処理を登録する。"""

        normalized = provider_type.strip().lower()
        if not normalized:
            raise ValueError("プロバイダー種類名を指定してください")
        self._builders[normalized] = builder

    def create(self, settings: ResolvedProviderSettings) -> ChatProvider:
        """解決済み設定からプロバイダーを作成する。"""

        normalized = settings.provider.strip().lower()
        builder = self._builders.get(normalized)
        if builder is None:
            raise ProviderConfigurationError(
                f"プロバイダー種類「{settings.provider}」には未対応です。"
            )
        return builder(settings)


def _build_mock(settings: ResolvedProviderSettings) -> ChatProvider:
    return MockProvider(name=settings.name, model=settings.model or "mock-model")


def _build_openai_compatible(settings: ResolvedProviderSettings) -> ChatProvider:
    return OpenAICompatibleProvider(
        name=settings.name,
        base_url=settings.base_url,
        api_key=settings.api_key,
        model=settings.model,
        temperature=settings.temperature,
        max_output_tokens=settings.max_output_tokens,
        timeout=settings.timeout,
    )


_DEFAULT_FACTORY = ProviderFactory()


def build_provider(settings: ResolvedProviderSettings) -> ChatProvider:
    """標準登録済みの実装からプロバイダーを作る。"""

    return _DEFAULT_FACTORY.create(settings)
