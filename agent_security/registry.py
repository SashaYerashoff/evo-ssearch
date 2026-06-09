from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock
from types import MappingProxyType
from typing import Any, Callable, Mapping

from .context import ToolExecutionContext
from .errors import ToolNotFoundError
from .policy import ToolPolicy


ToolHandler = Callable[
    [ToolExecutionContext, Mapping[str, Any]],
    Any,
]


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    name: str
    handler: ToolHandler
    policy: ToolPolicy
    description: str = ""
    input_schema: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        name = self.name.strip()
        if not name:
            raise ValueError("tool name is required")
        object.__setattr__(self, "name", name)
        object.__setattr__(
            self,
            "input_schema",
            MappingProxyType(dict(self.input_schema)),
        )


class ToolRegistry:
    def __init__(self) -> None:
        self._definitions: dict[str, ToolDefinition] = {}
        self._lock = RLock()

    def register(
        self,
        name: str | ToolDefinition,
        handler: ToolHandler | None = None,
        policy: ToolPolicy | None = None,
        *,
        description: str = "",
        input_schema: Mapping[str, Any] | None = None,
        replace: bool = False,
    ) -> ToolDefinition:
        if isinstance(handler, ToolPolicy) and callable(policy):
            handler, policy = policy, handler
        if isinstance(name, ToolDefinition):
            if handler is not None or policy is not None:
                raise TypeError(
                    "handler and policy are not accepted with ToolDefinition"
                )
            definition = name
        else:
            if handler is None or policy is None:
                raise TypeError("handler and policy are required")
            definition = ToolDefinition(
                name=name,
                handler=handler,
                policy=policy,
                description=description,
                input_schema=input_schema or {},
            )

        with self._lock:
            if definition.name in self._definitions and not replace:
                raise ValueError(f"tool already registered: {definition.name}")
            self._definitions[definition.name] = definition
        return definition

    def tool(
        self,
        name: str,
        policy: ToolPolicy,
        *,
        description: str = "",
        input_schema: Mapping[str, Any] | None = None,
    ) -> Callable[[ToolHandler], ToolHandler]:
        def decorator(handler: ToolHandler) -> ToolHandler:
            self.register(
                name,
                handler,
                policy,
                description=description,
                input_schema=input_schema,
            )
            return handler

        return decorator

    def get(self, name: str) -> ToolDefinition:
        with self._lock:
            definition = self._definitions.get(name)
        if definition is None:
            raise ToolNotFoundError(f"unknown tool: {name}")
        return definition

    def names(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._definitions))

    def definitions(self) -> tuple[ToolDefinition, ...]:
        with self._lock:
            return tuple(
                self._definitions[name] for name in sorted(self._definitions)
            )
