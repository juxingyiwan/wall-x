"""Validation helpers for serialized training configuration and metadata."""


def is_builtin_int(value) -> bool:
    """Reject booleans and integer subclasses where a built-in int is required."""
    return type(value) is int  # noqa: E721 - exact type is intentional.
