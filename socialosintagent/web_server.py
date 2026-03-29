"""
FastAPI web server for the OSINT agent web interface.

Provides:
  - REST API at /api/v1/ for session management, target CRUD, cache ops
  - Server-Sent Events at /api/v1/jobs/{job_id}/stream for live progress
  - Static file serving for the single-page frontend
  - /api/v1/platforms to list available configured platforms

The CLI interface is completely unchanged — this server is an additional
way to drive the same SocialOSINTAgent engine. Both interfaces share the
same data/ directory (cache, media, outputs).

Authentication: Basic HTTP auth via OSINT_WEB_USER / OSINT_WEB_PASSWORD
environment variables. If neither is set the server warns and runs open
(suitable for localhost-only SSH-tunnel access).

Run with:
    uvicorn socialosintagent.web_server:app --host 0.0.0.0 --port 8000
Or via docker-compose (see docker-compose.yml).
"""

import asyncio
import json
import logging
import os
import secrets
import shutil
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles

from .analyzer import SocialOSINTAgent
from .api_models import (
    AnalysisRequest,
    CacheStatusResponse,
    ContactsResponse,
    DiscoveredContactItem,
    DismissContactRequest,
    JobStatusResponse,
    PlatformInfo,
    PlatformsResponse,
    PurgeRequest,
    SessionCreateRequest,
    SessionRenameRequest,
    SessionUpdateTargetsRequest,
)
from .cache import CacheManager
from .client_manager import ClientManager
from .llm import LLMAnalyzer
from .network_extractor import extract_contacts as _extract_contacts
from .session_manager import SessionManager

load_dotenv()

# Configure logging so web server logs are saved to file just like CLI
logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(logs_dir / "analyzer.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("SocialOSINTAgent.WebServer")

# ---------------------------------------------------------------------------
# Application factory helpers
# ---------------------------------------------------------------------------

BASE_DIR = Path("data")
STATIC_DIR = Path(__file__).parent.parent / "static"

# Thread pool for running the synchronous agent in async context.
# Single worker ensures we don't make simultaneous multi-API calls that
# would exhaust rate limits. Can be increased if platforms allow it.
_EXECUTOR = ThreadPoolExecutor(max_workers=1)

# In-memory job registry: job_id -> job state dict.
# Jobs are transient — they exist only for the lifetime of the server process.
# Results are persisted to the session file once complete.
_JOBS: Dict[str, Dict[str, Any]] = {}

# ---------------------------------------------------------------------------
# FastAPI app setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="OSINT Agent Web API",
    version="1.0.0",
    description="Web interface for the OWASP Social OSINT Agent",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # Must be False when allow_origins=["*"].
    # Starlette/FastAPI rejects credentials=True + wildcard origin (violates CORS spec).
    # Basic Auth is sent as an Authorization header, not a cookie, so credentials=False
    # is correct here. Tighten allow_origins in production if exposed beyond localhost.
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Optional HTTP Basic Auth
# ---------------------------------------------------------------------------

_security = HTTPBasic(auto_error=False)
_WEB_USER = os.getenv("OSINT_WEB_USER", "")
_WEB_PASSWORD = os.getenv("OSINT_WEB_PASSWORD", "")

if not _WEB_USER or not _WEB_PASSWORD:
    logger.warning(
        "OSINT_WEB_USER / OSINT_WEB_PASSWORD not set. "
        "Web interface is running without authentication. "
        "Recommended: expose only via SSH tunnel or set credentials."
    )


def _check_auth(credentials: Optional[HTTPBasicCredentials] = Depends(_security)):
    """
    Validates HTTP Basic credentials if configured.
    If OSINT_WEB_USER/PASSWORD are not set, auth is skipped entirely.
    Uses constant-time comparison to prevent timing attacks.
    """
    if not _WEB_USER or not _WEB_PASSWORD:
        return  # Auth not configured — open access
    
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Basic"},
        )
    
    correct_user = secrets.compare_digest(credentials.username.encode(), _WEB_USER.encode())
    correct_pass = secrets.compare_digest(credentials.password.encode(), _WEB_PASSWORD.encode())
    
    if not (correct_user and correct_pass):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )


# ---------------------------------------------------------------------------
# Dependency: shared agent components
# ---------------------------------------------------------------------------

def _get_components():
    """
    Builds and returns the shared agent components.
    Called once per request where needed — components are lightweight to
    construct since the LLM client and API clients are lazily initialised.
    """
    cache_manager = CacheManager(BASE_DIR, is_offline=False)
    llm_analyzer = LLMAnalyzer(is_offline=False)
    client_manager = ClientManager(is_offline=False)
    session_manager = SessionManager(BASE_DIR)
    return cache_manager, llm_analyzer, client_manager, session_manager


# ---------------------------------------------------------------------------
# SSE progress event helpers
# ---------------------------------------------------------------------------

def _make_event(event_type: str, data: Dict[str, Any]) -> str:
    """Formats a Server-Sent Event string."""
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


def _push_progress(job_id: str, event_type: str, data: Dict[str, Any]):
    """
    Appends a progress event to the job's event queue.
    Also updates the job's latest progress snapshot for polling clients.
    """
    if job_id not in _JOBS:
        return
    event = {"type": event_type, "data": data, "ts": datetime.now(timezone.utc).isoformat()}
    _JOBS[job_id]["events"].append(event)
    _JOBS[job_id]["progress"] = data


# ---------------------------------------------------------------------------
# Analysis background job
# ---------------------------------------------------------------------------

def _run_analysis_job(
    job_id: str,
    session_id: str,
    query: str,
    force_refresh: bool,
    platforms: Dict[str, List[str]],
    fetch_options: Dict[str, Any],
):
    """
    Runs the full analysis pipeline in a thread pool worker.

    Progress is pushed into the job's event queue as each stage completes.
    The job state is updated to 'complete' or 'error' when done.
    Results are persisted to the session file.

    This function is intentionally synchronous — it runs in a ThreadPoolExecutor
    so the async FastAPI event loop is never blocked.
    """
    import argparse

    # Build a minimal args namespace matching what SocialOSINTAgent expects
    args = argparse.Namespace(
        offline=False,
        no_auto_save=True,
        format="markdown",
        unsafe_allow_external_media=False,
    )

    try:
        cache_manager, llm_analyzer, client_manager, session_manager = _get_components()

        agent = SocialOSINTAgent(args, cache_manager, llm_analyzer, client_manager)

        _push_progress(job_id, "stage", {"stage": "fetch", "message": "Fetching platform data..."})

        result = agent.analyze(
            platforms,
            query,
            force_refresh,
            fetch_options,
        )

        if result.get("error"):
            _JOBS[job_id]["status"] = "error"
            _JOBS[job_id]["error"] = result.get("report", "Analysis failed")
            _push_progress(job_id, "error", {"message": result.get("report", "Analysis failed")})
            return

        # Persist result to session
        session = session_manager.load(session_id)
        if session:
            query_id = session.add_query_result(
                query=query,
                report=result["report"],
                metadata=result.get("metadata", {}),
                entities=result.get("entities", {}),
            )
            session_manager.save(session)
            _JOBS[job_id]["query_id"] = query_id

            # Automatically save the markdown report to the outputs directory
            try:
                agent._save_output_headless(result, args.format)
            except Exception as e:
                logger.error(f"Failed to save standalone markdown report: {e}")

        _JOBS[job_id]["status"] = "complete"
        _push_progress(job_id, "complete", {
            "message": "Analysis complete",
            "query_id": _JOBS[job_id].get("query_id"),
        })

    except Exception as e:
        logger.error(f"Analysis job {job_id} failed: {e}", exc_info=True)
        _JOBS[job_id]["status"] = "error"
        _JOBS[job_id]["error"] = str(e)
        _push_progress(job_id, "error", {"message": str(e)})


# ---------------------------------------------------------------------------
# API routes — v1
# ---------------------------------------------------------------------------

# ---- Platforms ----

@app.get(
    "/api/v1/platforms",
    response_model=PlatformsResponse,
    summary="List available configured platforms",
    dependencies=[Depends(_check_auth)],
)
def get_platforms():
    """Returns which platforms have credentials configured and are available."""
    _, _, client_manager, _ = _get_components()
    all_known = ["twitter", "reddit", "bluesky", "mastodon", "github", "hackernews"]
    try:
        # get_available_platforms(check_creds=True) is confirmed in ClientManager
        # from cli_handler.py — returns a list of platform name strings.
        available = client_manager.get_available_platforms(check_creds=True)
    except Exception as e:
        logger.warning(f"Could not determine platform availability: {e}")
        available = []
    platforms = [
        PlatformInfo(name=p, available=(p in available))
        for p in all_known
    ]
    return PlatformsResponse(platforms=platforms)


# ---- Sessions ----

@app.get(
    "/api/v1/sessions",
    summary="List all sessions",
    dependencies=[Depends(_check_auth)],
)
def list_sessions():
    """Returns summary info for all sessions, sorted by most recently updated."""
    _, _, _, session_manager = _get_components()
    return {"sessions": session_manager.list_all()}


@app.post(
    "/api/v1/sessions",
    status_code=201,
    summary="Create a new session",
    dependencies=[Depends(_check_auth)],
)
def create_session(body: SessionCreateRequest):
    """Creates a new named session with an initial set of targets."""
    _, _, _, session_manager = _get_components()
    session = session_manager.create(
        name=body.name,
        platforms=body.platforms,
        fetch_options=body.fetch_options,
    )
    return session.to_dict()


@app.get(
    "/api/v1/sessions/{session_id}",
    summary="Get a session (full, including query history)",
    dependencies=[Depends(_check_auth)],
)
def get_session(session_id: str):
    """Returns the full session including all past query results."""
    _, _, _, session_manager = _get_components()
    session = session_manager.load(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session.to_dict()


@app.delete(
    "/api/v1/sessions/{session_id}",
    summary="Delete a session",
    dependencies=[Depends(_check_auth)],
)
def delete_session(session_id: str):
    """Permanently deletes a session and its query history."""
    _, _, _, session_manager = _get_components()
    if not session_manager.delete(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    return {"deleted": session_id}


@app.patch(
    "/api/v1/sessions/{session_id}/rename",
    summary="Rename a session",
    dependencies=[Depends(_check_auth)],
)
def rename_session(session_id: str, body: SessionRenameRequest):
    """Updates the human-readable name of an existing session."""
    _, _, _, session_manager = _get_components()
    session = session_manager.rename(session_id, body.name)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session.summary()


@app.put(
    "/api/v1/sessions/{session_id}/targets",
    summary="Update session targets",
    dependencies=[Depends(_check_auth)],
)
def update_targets(session_id: str, body: SessionUpdateTargetsRequest):
    """
    Replaces the target list and fetch options for an existing session.
    This is the /add and /remove equivalent for the web interface.
    """
    _, _, _, session_manager = _get_components()
    session = session_manager.update_targets(
        session_id,
        platforms=body.platforms,
        fetch_options=body.fetch_options,
    )
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session.summary()


# ---- Analysis jobs ----

@app.post(
    "/api/v1/sessions/{session_id}/analyse",
    status_code=202,
    summary="Start an analysis job",
    dependencies=[Depends(_check_auth)],
)
async def start_analysis(
    session_id: str,
    body: AnalysisRequest,
):
    """
    Starts an analysis job for the session.

    Returns immediately with a job_id. Use GET /api/v1/jobs/{job_id}
    to poll status, or GET /api/v1/jobs/{job_id}/stream for SSE progress.
    """
    _, _, _, session_manager = _get_components()
    session = session_manager.load(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Check for an already-running job for this session
    for job in _JOBS.values():
        if job["session_id"] == session_id and job["status"] == "running":
            raise HTTPException(
                status_code=409,
                detail=f"Session already has a running analysis job: {job['job_id']}",
            )

    job_id = str(uuid.uuid4())
    _JOBS[job_id] = {
        "job_id": job_id,
        "session_id": session_id,
        "status": "running",
        "query": body.query,
        "query_id": None,
        "error": None,
        "progress": None,
        "events": [],
    }

    # Run the synchronous agent in the thread pool — never blocks the event loop.
    # asyncio.get_running_loop() is correct here (Python 3.10+): we are inside an
    # async route handler so there is always a running loop. get_event_loop() is
    # deprecated in async contexts and will raise a DeprecationWarning in 3.10+.
    loop = asyncio.get_running_loop()
    loop.run_in_executor(
        _EXECUTOR,
        _run_analysis_job,
        job_id,
        session_id,
        body.query,
        body.force_refresh,
        session.platforms,
        session.fetch_options,
    )

    return {"job_id": job_id, "session_id": session_id, "status": "running"}


@app.get(
    "/api/v1/jobs/{job_id}",
    response_model=JobStatusResponse,
    summary="Poll job status",
    dependencies=[Depends(_check_auth)],
)
def get_job_status(job_id: str):
    """Returns the current status of an analysis job. Suitable for polling."""
    if job_id not in _JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    job = _JOBS[job_id]
    return JobStatusResponse(
        job_id=job_id,
        session_id=job["session_id"],
        status=job["status"],
        query=job["query"],
        query_id=job.get("query_id"),
        error=job.get("error"),
        progress=job.get("progress"),
    )


@app.get(
    "/api/v1/jobs/{job_id}/stream",
    summary="Stream job progress via SSE",
    dependencies=[Depends(_check_auth)],
)
async def stream_job_progress(job_id: str):
    """
    Server-Sent Events endpoint for live progress during an analysis job.

    The client receives events as the analysis progresses:
      - 'stage'    — major pipeline stage started (fetch, vision, synthesis)
      - 'log'      — informational message from the agent
      - 'status'   — spinner-style status update
      - 'complete' — job finished successfully, includes query_id
      - 'error'    — job failed, includes error message

    Replays all events emitted so far on connection, so late-connecting
    clients (e.g. page reload mid-analysis) catch up automatically.
    """
    if job_id not in _JOBS:
        raise HTTPException(status_code=404, detail="Job not found")

    async def event_generator() -> AsyncGenerator[str, None]:
        job = _JOBS[job_id]
        sent_index = 0

        # Replay any events already emitted (handles late connect / page reload)
        while True:
            events = job.get("events", [])
            while sent_index < len(events):
                ev = events[sent_index]
                yield _make_event(ev["type"], ev["data"])
                sent_index += 1

            # If job is done and we've sent everything, close the stream
            if job["status"] in ("complete", "error") and sent_index >= len(job.get("events", [])):
                break

            # Yield a keepalive comment every second while waiting for more events
            yield ": keepalive\n\n"
            await asyncio.sleep(1)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Disable nginx buffering for SSE
        },
    )


# ---- Cache management ----

@app.get(
    "/api/v1/cache",
    response_model=CacheStatusResponse,
    summary="Get cache status",
    dependencies=[Depends(_check_auth)],
)
def get_cache_status():
    """
    Returns a summary of all cached platform data: platform, username,
    post count, media counts, cache age, and freshness status.
    """
    import json as _json
    from .cache import CACHE_EXPIRY_HOURS

    cache_dir = BASE_DIR / "cache"
    entries = []

    if cache_dir.is_dir():
        for path in sorted(cache_dir.glob("*.json")):
            try:
                parts = path.stem.split("_", 1)
                if len(parts) != 2:
                    continue
                platform, username = parts
                data = _json.loads(path.read_text(encoding="utf-8"))

                from .utils import get_sort_key

                ts = get_sort_key(data, "timestamp")
                age_seconds = (datetime.now(timezone.utc) - ts).total_seconds()
                is_fresh = age_seconds < CACHE_EXPIRY_HOURS * 3600

                media_found = sum(
                    len(post.get("media", []))
                    for post in data.get("posts", [])
                )
                media_analyzed = sum(
                    1
                    for post in data.get("posts", [])
                    for m in post.get("media", [])
                    if m.get("analysis")
                )

                entries.append({
                    "platform": platform,
                    "username": data.get("profile", {}).get("username", username),
                    "post_count": len(data.get("posts", [])),
                    "media_found": media_found,
                    "media_analyzed": media_analyzed,
                    "cached_at": ts.isoformat(),
                    "age_seconds": int(age_seconds),
                    "is_fresh": is_fresh,
                })
            except Exception as e:
                logger.warning(f"Could not read cache file {path.name}: {e}")

    return CacheStatusResponse(entries=entries)


@app.post(
    "/api/v1/cache/purge",
    summary="Purge cached data",
    dependencies=[Depends(_check_auth)],
)
def purge_cache(body: PurgeRequest):
    """
    Purges selected data directories or specific keys.
    """
    cm, _, _, _ = _get_components()
    if body.keys:
        purged_keys = []
        for key in body.keys:
            if "_" in key:
                platform, username = key.split("_", 1)
                cm.delete(platform, username)
                purged_keys.append(key)
        return {"purged": purged_keys}

    targets = body.targets
    if "all" in targets:
        targets = ["cache", "media", "outputs"]

    purged = []
    for target in targets:
        path = BASE_DIR / target
        if path.exists():
            shutil.rmtree(path)
            purged.append(target)
        path.mkdir(parents=True, exist_ok=True)

    return {"purged": purged}


# ---- Network / contacts ----

@app.get(
    "/api/v1/sessions/{session_id}/contacts",
    response_model=ContactsResponse,
    summary="Get discovered network contacts for a session",
    dependencies=[Depends(_check_auth)],
)
def get_session_contacts(session_id: str):
    """
    Returns contacts discovered from the cached posts of all active targets in
    the session — @mentions, retweets, repo interactions, etc.

    Operates entirely on locally cached post data; no API calls are made.
    Active targets and previously dismissed accounts are excluded from the
    returned list. Results are sorted by weight (most-interacted-with first).
    """
    cache_manager, _, _, session_manager = _get_components()
    session = session_manager.load(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    # Build posts dict from cache — one entry per (platform, username)
    platform_posts: dict = {}
    for platform, usernames in session.platforms.items():
        platform_posts[platform] = {}
        for username in usernames:
            data = cache_manager.load(platform, username)
            if data:
                platform_posts[platform][username] = data.get("posts", [])

    all_contacts = _extract_contacts(platform_posts, session.platforms)
    dismissed_set = set(session.dismissed_contacts)

    filtered = [
        c for c in all_contacts
        if f"{c.platform}/{c.username.lower()}" not in dismissed_set
    ]

    contact_items = [
        DiscoveredContactItem(
            platform=c.platform,
            username=c.username,
            interaction_types=sorted(set(c.interaction_types)),
            weight=c.weight,
            first_seen=c.first_seen.isoformat() if c.first_seen else None,
            last_seen=c.last_seen.isoformat() if c.last_seen else None,
        )
        for c in filtered
    ]

    return ContactsResponse(
        contacts=contact_items,
        dismissed=list(dismissed_set),
        total_extracted=len(all_contacts),
    )


@app.post(
    "/api/v1/sessions/{session_id}/contacts/dismiss",
    summary="Dismiss a discovered contact",
    dependencies=[Depends(_check_auth)],
)
def dismiss_contact(session_id: str, body: DismissContactRequest):
    """
    Marks a contact as dismissed so it is hidden from the network panel on
    future GET /contacts calls. Persisted to the session file.
    Use the undismiss endpoint to reverse this.
    """
    _, _, _, session_manager = _get_components()
    session = session_manager.dismiss_contact(session_id, body.platform, body.username)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return {"dismissed": f"{body.platform}/{body.username.lower()}"}


@app.post(
    "/api/v1/sessions/{session_id}/contacts/undismiss",
    summary="Restore a previously dismissed contact",
    dependencies=[Depends(_check_auth)],
)
def undismiss_contact(session_id: str, body: DismissContactRequest):
    """
    Removes a contact from the session's dismissed list, making it visible
    in the network panel again.
    """
    _, _, _, session_manager = _get_components()
    session = session_manager.undismiss_contact(session_id, body.platform, body.username)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return {"undismissed": f"{body.platform}/{body.username.lower()}"}


# ---- OSINT Specific APIs (Export, Timeline, Media) ----

@app.get(
    "/api/v1/sessions/{session_id}/export",
    summary="Export full session OSINT data",
    dependencies=[Depends(_check_auth)],
)
def export_session(session_id: str):
    """Exports session data including queries, entities, and contacts."""
    cache_manager, _, _, session_manager = _get_components()
    session = session_manager.load(session_id)
    if not session: raise HTTPException(status_code=404)
    
    # Bundle contacts
    platform_posts = {}
    for platform, usernames in session.platforms.items():
        platform_posts[platform] = {}
        for user in usernames:
            if data := cache_manager.load(platform, user):
                platform_posts[platform][user] = data.get("posts", [])
    contacts = [c.to_dict() for c in _extract_contacts(platform_posts, session.platforms)]
    
    export_data = session.to_dict()
    export_data["extracted_network"] = contacts
    
    return JSONResponse(
        content=export_data,
        headers={"Content-Disposition": f'attachment; filename="osint_export_{session_id}.json"'}
    )

@app.get(
    "/api/v1/sessions/{session_id}/timeline",
    summary="Get timeline events for heatmap",
    dependencies=[Depends(_check_auth)],
)
def get_timeline(session_id: str):
    """Returns all post timestamps to generate a pattern-of-life heatmap."""
    cache_manager, _, _, session_manager = _get_components()
    session = session_manager.load(session_id)
    if not session: raise HTTPException(status_code=404)
    
    events = []
    for platform, usernames in session.platforms.items():
        for user in usernames:
            if data := cache_manager.load(platform, user):
                for post in data.get("posts", []):
                    ts = post.get("created_at")
                    if ts:
                        events.append({"timestamp": ts, "platform": platform, "author": user})
    return {"events": events}

@app.get(
    "/api/v1/sessions/{session_id}/media",
    summary="Get session media paths",
    dependencies=[Depends(_check_auth)],
)
def get_media(session_id: str):
    """Returns all downloaded media and their associated LLM analysis."""
    cache_manager, _, _, session_manager = _get_components()
    session = session_manager.load(session_id)
    if not session: raise HTTPException(status_code=404)
    
    media_items = []
    for platform, usernames in session.platforms.items():
        for user in usernames:
            if data := cache_manager.load(platform, user):
                for post in data.get("posts", []):
                    for m in post.get("media", []):
                        # Ensure we only return successfully downloaded local images
                        if m.get("local_path"):
                            media_items.append({
                                "url": m.get("url"),
                                "path": m.get("local_path"),
                                "analysis": m.get("analysis", ""),
                                "post_id": post.get("id"),
                                "platform": platform,
                                "author": user
                            })
    return {"media": media_items}

@app.get(
    "/api/v1/sessions/{session_id}/media/file",
    summary="Serve local media file",
    dependencies=[Depends(_check_auth)],
)
def get_media_file(session_id: str, path: str):
    """Serves the actual image bytes to the frontend."""
    # Security: ensure the requested path is actually inside our media directory
    requested_path = Path(path).resolve()
    media_dir = (BASE_DIR / "media").resolve()
    if media_dir not in requested_path.parents:
        raise HTTPException(status_code=403, detail="Forbidden")
    if not requested_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(requested_path)


# ---- Static files (frontend) ----
# Served last so API routes take priority over any static file with the same path.

if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
else:
    @app.get("/")
    def root():
        return JSONResponse(
            {"message": "OSINT Agent API is running. Static frontend not found."},
            status_code=200,
        )


# ---------------------------------------------------------------------------
# Entry point for direct execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "socialosintagent.web_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )