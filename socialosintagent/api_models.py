"""
Pydantic request/response models for the FastAPI web server.

All API input is validated through these models before reaching the agent.
Keeping models in a separate file makes them easy to version alongside the
/api/v1/ route prefix — if the API shape changes, add v2 models here.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Session models
# ---------------------------------------------------------------------------

class SessionCreateRequest(BaseModel):
    """Request body for POST /api/v1/sessions"""
    name: str = Field(..., min_length=1, max_length=100, description="Human-readable session name")
    platforms: Dict[str, List[str]] = Field(..., description="Platform -> list of usernames")
    fetch_options: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Fetch options: default_count and per-target overrides"
    )

    @field_validator("platforms")
    @classmethod
    def platforms_not_empty(cls, v):
        if not v:
            raise ValueError("platforms must contain at least one entry")
        for platform, usernames in v.items():
            if not usernames:
                raise ValueError(f"Platform '{platform}' has no usernames")
        return v


class SessionRenameRequest(BaseModel):
    """Request body for PATCH /api/v1/sessions/{session_id}/rename"""
    name: str = Field(..., min_length=1, max_length=100)


class SessionUpdateTargetsRequest(BaseModel):
    """Request body for PUT /api/v1/sessions/{session_id}/targets"""
    platforms: Dict[str, List[str]] = Field(..., description="New complete platforms dict")
    fetch_options: Optional[Dict[str, Any]] = Field(default=None)

    @field_validator("platforms")
    @classmethod
    def platforms_not_empty(cls, v):
        if not v:
            raise ValueError("platforms must contain at least one entry")
        return v


# ---------------------------------------------------------------------------
# Analysis models
# ---------------------------------------------------------------------------

class AnalysisRequest(BaseModel):
    """Request body for POST /api/v1/sessions/{session_id}/analyse"""
    query: str = Field(..., min_length=1, max_length=500, description="Natural language analysis query")
    force_refresh: bool = Field(
        default=False,
        description="If true, bypass the 24h cache and re-fetch all platform data"
    )


class JobStatusResponse(BaseModel):
    """Response for job status polling — GET /api/v1/jobs/{job_id}"""
    job_id: str
    session_id: str
    status: str  # "running" | "complete" | "error"
    query: str
    query_id: Optional[str] = None   # Set when complete
    error: Optional[str] = None      # Set on error
    progress: Optional[Dict[str, Any]] = None  # Latest progress snapshot


# ---------------------------------------------------------------------------
# Cache models
# ---------------------------------------------------------------------------

class CacheStatusResponse(BaseModel):
    """Response for GET /api/v1/cache"""
    entries: List[Dict[str, Any]]


class PurgeRequest(BaseModel):
    """Request body for POST /api/v1/cache/purge"""
    targets: List[str] = Field(
        ...,
        description="Which data types to purge: 'cache', 'media', 'outputs', 'all', or 'specific'"
    )
    keys: Optional[List[str]] = Field(
        default=None, 
        description="Specific cache keys to delete e.g. ['twitter_user1']"
    )

    @field_validator("targets")
    @classmethod
    def valid_targets(cls, v):
        allowed = {"cache", "media", "outputs", "all", "specific"}
        for t in v:
            if t not in allowed:
                raise ValueError(f"Invalid purge target '{t}'. Allowed: {allowed}")
        return v


# ---------------------------------------------------------------------------
# Platform info models
# ---------------------------------------------------------------------------

class PlatformInfo(BaseModel):
    """Single platform availability entry."""
    name: str
    available: bool
    reason: Optional[str] = None  # Why unavailable, if applicable


class PlatformsResponse(BaseModel):
    """Response for GET /api/v1/platforms"""
    platforms: List[PlatformInfo]


# ---------------------------------------------------------------------------
# Network / contact models
# ---------------------------------------------------------------------------

class DiscoveredContactItem(BaseModel):
    """
    A single contact discovered from a target user's posts.

    Returned as part of ContactsResponse. The interaction_types list records
    every distinct way this contact was found (mention, retweet, repo_interaction,
    etc.), and weight is the total number of times any interaction was seen.
    """
    platform: str
    username: str
    interaction_types: List[str]
    weight: int
    first_seen: Optional[str] = None  # ISO 8601 timestamp or None
    last_seen: Optional[str] = None   # ISO 8601 timestamp or None


class ContactsResponse(BaseModel):
    """
    Response for GET /api/v1/sessions/{session_id}/contacts.

    contacts:        Discovered contacts, sorted by weight descending, with
                     currently-active targets and dismissed entries excluded.
    dismissed:       The session's current "platform/username" dismiss list,
                     so the UI can restore dismissed entries if needed.
    total_extracted: Total contacts found before filtering — useful for the
                     UI to show e.g. "12 contacts (3 dismissed)".
    """
    contacts: List[DiscoveredContactItem]
    dismissed: List[str]
    total_extracted: int


class DismissContactRequest(BaseModel):
    """
    Request body for POST /api/v1/sessions/{session_id}/contacts/dismiss
    and POST /api/v1/sessions/{session_id}/contacts/undismiss.
    """
    platform: str = Field(..., min_length=1, description="Platform the contact was found on")
    username: str = Field(..., min_length=1, description="Contact's username")


# ---------------------------------------------------------------------------
# Error model
# ---------------------------------------------------------------------------

class ErrorResponse(BaseModel):
    """Standard error response envelope."""
    error: str
    detail: Optional[str] = None