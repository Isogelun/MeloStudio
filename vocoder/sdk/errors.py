class VocoderSDKError(Exception):
    """Base exception for stable SDK error handling."""


class InvalidFeatureShape(VocoderSDKError, ValueError):
    pass


class FeatureValueError(VocoderSDKError, ValueError):
    pass


class FeatureConfigMismatch(VocoderSDKError, ValueError):
    pass


class CheckpointCompatibilityError(VocoderSDKError, RuntimeError):
    pass


class UnsupportedInputType(VocoderSDKError, TypeError):
    pass
