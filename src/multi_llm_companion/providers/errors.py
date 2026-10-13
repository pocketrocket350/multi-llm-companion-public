"""画面へ安全な日本語で表示できるプロバイダー例外。"""


class ProviderError(RuntimeError):
    """プロバイダー処理に失敗したときの基底例外。"""


class ProviderConfigurationError(ProviderError):
    """接続前に見つかった設定不足。"""


class ProviderConnectionError(ProviderError):
    """タイムアウトやネットワーク切断。"""


class ProviderAuthenticationError(ProviderError):
    """APIキーが拒否された状態。"""


class ProviderResponseError(ProviderError):
    """APIが想定外の応答を返した状態。"""


class _EmptyProviderResponse(ProviderResponseError):
    """通信は完了したが、返答の本文を得られなかった。"""
