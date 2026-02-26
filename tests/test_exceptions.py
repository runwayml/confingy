"""
Tests for confingy.exceptions module - error handling and validation.
"""

import pytest

from confingy import ValidationError, track
from confingy.exceptions import DeserializationError, SerializationError
from tests.conftest import Adder


def test_validation_error():
    """Test ValidationError with improved debugging context."""

    with pytest.raises(ValidationError) as exc_info:
        Adder(amount="not a number")

    error = exc_info.value

    # Check that the error contains the class name
    assert "Adder" in str(error)

    # Check that the error contains field information
    assert "amount" in str(error)

    # Check that the error contains the invalid value
    assert "not a number" in str(error)

    # Check that config information is included
    assert "Provided configuration:" in str(error)


def test_validation_error_preserves_original():
    """Test that ValidationError preserves the original Pydantic error."""

    try:
        Adder(amount="invalid")
    except ValidationError as e:
        # Should preserve the original Pydantic error
        assert hasattr(e, "error")
        assert hasattr(e, "cls_name")
        assert hasattr(e, "config")

        assert e.cls_name == "Adder"
        assert e.config == {"amount": "invalid"}


def test_validation_error_without_config():
    """Test ValidationError when config is not provided."""
    from pydantic import ValidationError as PydanticValidationError

    from confingy.exceptions import ValidationError

    # Create a mock Pydantic error
    try:
        from pydantic import BaseModel

        class TestModel(BaseModel):
            value: int

        TestModel(value="not_an_int")
    except PydanticValidationError as pydantic_error:
        # Create ValidationError without config
        error = ValidationError(pydantic_error, "TestClass")

        # Should still work without config
        assert "TestClass" in str(error)
        assert "Provided configuration:" not in str(error)


def test_validation_error_multiple_fields():
    """Test ValidationError with multiple field errors."""

    @track
    class MultiFieldClass:
        def __init__(self, field1: int, field2: str, field3: float):
            self.field1 = field1
            self.field2 = field2
            self.field3 = field3

    with pytest.raises(ValidationError) as exc_info:
        MultiFieldClass(field1="not_int", field2=123, field3="not_float")

    error_msg = str(exc_info.value)

    # Should contain information about all failing fields
    assert "field1" in error_msg
    assert (
        "field2" in error_msg or "field3" in error_msg
    )  # At least one of the other fields

    # Should contain the provided configuration
    assert "Provided configuration:" in error_msg
    assert "not_int" in error_msg
    assert "123" in error_msg
    assert "not_float" in error_msg


def test_validation_error_nested_field():
    """Test ValidationError with nested field paths."""

    @track
    class NestedClass:
        def __init__(self, nested_dict: dict[str, int]):
            self.nested_dict = nested_dict

    # This should cause a validation error due to wrong types in the dict
    try:
        NestedClass(nested_dict={"valid": 1, "invalid": "not_an_int"})
    except ValidationError as e:
        # Should work even with complex nested validation
        assert "NestedClass" in str(e)


def test_error_message_formatting():
    """Test that error messages are well-formatted and readable."""

    @track
    class WellDocumentedClass:
        """A well-documented class for testing."""

        def __init__(self, important_param: int, optional_param: str = "default"):
            self.important_param = important_param
            self.optional_param = optional_param

    with pytest.raises(ValidationError) as exc_info:
        WellDocumentedClass(important_param="wrong_type")

    error_msg = str(exc_info.value)

    # Should be well-formatted with bullet points
    assert "•" in error_msg or "*" in error_msg

    # Should separate sections clearly
    assert "\n" in error_msg

    # Should show both the error and the config
    lines = error_msg.split("\n")
    assert len(lines) > 3  # Should be multi-line for better readability


def test_validation_error_repr():
    """Test that ValidationError has a reasonable repr."""

    try:
        Adder(amount="invalid")
    except ValidationError as e:
        repr_str = repr(e)
        assert "ValidationError" in repr_str
        assert "Adder" in repr_str


def test_chained_exceptions():
    """Test that exceptions can be chained properly."""

    # Test that our custom exceptions work well with exception chaining
    try:
        raise SerializationError("Original error")
    except SerializationError as e:
        try:
            raise DeserializationError("Chained error") from e
        except DeserializationError as chained:
            assert chained.__cause__ is e
            assert isinstance(chained.__cause__, SerializationError)
