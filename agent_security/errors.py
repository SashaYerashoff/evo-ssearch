class ToolGatewayError(Exception):
    """Base class for errors safe to expose at the tool boundary."""

    code = "tool_gateway_error"

    def __init__(self, message: str, *, details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ToolNotFoundError(ToolGatewayError):
    code = "tool_not_found"


class AuthorizationError(ToolGatewayError):
    code = "authorization_denied"


class PermissionDeniedError(AuthorizationError):
    code = "permission_denied"


class ChannelAccessDeniedError(AuthorizationError):
    code = "channel_access_denied"


class ContextInjectionError(AuthorizationError):
    code = "context_injection"


class InvalidToolArgumentsError(ToolGatewayError):
    code = "invalid_tool_arguments"


class ApprovalError(AuthorizationError):
    code = "invalid_approval"


class ApprovalRequiredError(ApprovalError):
    code = "approval_required"


class ApprovalArgumentMismatchError(ApprovalError):
    code = "approval_argument_mismatch"


class ApprovalExpiredError(ApprovalError):
    code = "approval_expired"


class ApprovalConsumedError(ApprovalError):
    code = "approval_consumed"


class PlanExpiredError(ApprovalError):
    code = "plan_expired"


class RateLimitExceededError(AuthorizationError):
    code = "rate_limit_exceeded"


class ToolTimeoutError(ToolGatewayError):
    code = "tool_timeout"


class AuditUnavailableError(ToolGatewayError):
    code = "audit_unavailable"
