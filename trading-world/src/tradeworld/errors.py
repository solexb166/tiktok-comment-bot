class TradeWorldError(Exception):
    """Base class for simulator errors."""


class ConfigError(TradeWorldError):
    """Invalid or inconsistent configuration."""


class UnverifiedSpecError(ConfigError):
    """A profile needs broker specifications that are not verified (environment B guard)."""

    def __init__(self, profile: str, fields: dict[str, str]):
        self.profile = profile
        self.fields = fields
        lines = "\n".join(f"  - {k}: {v}" for k, v in sorted(fields.items()))
        super().__init__(
            f"Profile {profile!r} requires verified specifications, but these are not verified:\n{lines}\n"
            "Resolve them in docs/exness-spec-verification.md, or run explicitly with sensitivity=True "
            "(results are tagged 'unverified-spec').")


class DataQualityError(TradeWorldError):
    """Market data failed validation."""
