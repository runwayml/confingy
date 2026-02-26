from typing import Optional

from pydantic import ValidationError as PydanticValidationError


class _BaseConfingyError(Exception):
    """Base exception for confingy errors."""

    pass


class ValidationError(_BaseConfingyError):
    """Raised when validation fails."""

    def __init__(
        self,
        error: PydanticValidationError,
        cls_name: str,
        config: Optional[dict] = None,
    ):
        self.error = error
        self.cls_name = cls_name
        self.config = config

        # Format a clear error message
        details = []
        for err in self.error.errors():
            loc = ".".join(str(loc_part) for loc_part in err.get("loc", []))
            msg = err.get("msg", "")
            input_value = err.get("input", "")

            if loc:
                details.append(f"  • Field '{loc}': {msg} (got {input_value!r})")
            else:
                details.append(f"  • {msg}")

        error_text = f"Validation failed for {cls_name}:\n" + "\n".join(details)

        # Add config information if available for debugging
        if config:
            config_str = "\n".join(f"  {k}: {v!r}" for k, v in config.items())
            error_text += f"\n\nProvided configuration:\n{config_str}"

        self.message = error_text
        super().__init__(self.message)


class SerializationError(_BaseConfingyError):
    """Raised when serialization fails."""

    pass


class DeserializationError(_BaseConfingyError):
    """Raised when deserialization fails."""

    pass
