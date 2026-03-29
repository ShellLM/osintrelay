class RateLimitExceededError(Exception):
    """Custom exception for API rate limits."""
    def __init__(self, message, original_exception=None):
        super().__init__(message)
        self.original_exception = original_exception


class UserNotFoundError(Exception):
    """Custom exception for when a user/profile cannot be found."""
    pass


class AccessForbiddenError(Exception):
    """Custom exception for access denied (e.g., private account)."""
    pass