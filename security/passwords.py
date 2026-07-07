from __future__ import annotations

from typing import Protocol, runtime_checkable


class PasswordHashingUnavailable(RuntimeError):
    """Raised when the required Argon2id implementation is unavailable."""


@runtime_checkable
class PasswordHasher(Protocol):
    def hash(self, password: str) -> str:
        ...

    def verify(self, password_hash: str, password: str) -> bool:
        ...

    def needs_rehash(self, password_hash: str) -> bool:
        ...


class Argon2idPasswordHasher:
    def __init__(
        self,
        *,
        time_cost: int = 3,
        memory_cost: int = 65536,
        parallelism: int = 4,
        hash_len: int = 32,
        salt_len: int = 16,
    ) -> None:
        try:
            from argon2 import PasswordHasher as Argon2PasswordHasher
            from argon2 import exceptions as argon2_exceptions
            from argon2.low_level import Type
        except ImportError as exc:
            raise PasswordHashingUnavailable(
                "Argon2id password hashing requires the 'argon2-cffi' package"
            ) from exc

        verification_errors = {argon2_exceptions.VerificationError}
        for exception_name in (
            "InvalidHash",
            "InvalidHashError",
            "VerifyMismatchError",
        ):
            exception_type = getattr(argon2_exceptions, exception_name, None)
            if isinstance(exception_type, type):
                verification_errors.add(exception_type)
        self._verification_errors = tuple(verification_errors)
        self._hasher = Argon2PasswordHasher(
            time_cost=time_cost,
            memory_cost=memory_cost,
            parallelism=parallelism,
            hash_len=hash_len,
            salt_len=salt_len,
            type=Type.ID,
        )

    def hash(self, password: str) -> str:
        if not password:
            raise ValueError("password is required")
        return str(self._hasher.hash(password))

    def verify(self, password_hash: str, password: str) -> bool:
        if not password_hash or not password:
            return False
        try:
            return bool(self._hasher.verify(password_hash, password))
        except self._verification_errors:
            return False

    def needs_rehash(self, password_hash: str) -> bool:
        if not password_hash:
            return True
        try:
            return bool(self._hasher.check_needs_rehash(password_hash))
        except self._verification_errors:
            return True


def create_password_hasher() -> PasswordHasher:
    return Argon2idPasswordHasher()
