class Deepstar3DError(RuntimeError):
    """Base expected pipeline error."""


class RightsConfirmationRequired(Deepstar3DError):
    pass


class AppleResearchLicenseRequired(Deepstar3DError):
    pass


class TrainingActive(Deepstar3DError):
    pass


class BackendUnavailable(Deepstar3DError):
    pass


class ImageGenerationFailed(Deepstar3DError):
    pass

