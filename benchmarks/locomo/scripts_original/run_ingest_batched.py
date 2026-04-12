#!/usr/bin/env python3
"""
LOCOMO batched concurrent ingest script

Uses a 3-3-4 batch strategy:
- Batch 1: 3 datasets in parallel
- Batch 2: 3 datasets in parallel
- Batch 3: 4 datasets in parallel

Each batch waits for all pipelines to finish before starting the next, to avoid:
- LLM rate limit overload
- SQLite lock contention
- Memory pressure

Time handling:
- Parse session timestamps (e.g. "1:56 pm on 8 May, 2023") to ISO 8601
- Add a time prefix to each message so mflow can parse relative time expressions correctly
- Pass session timestamp via API created_at as anchor time

Usage:
    python run_ingest_batched.py                    # prune DB then ingest
    python run_ingest_batched.py --no-prune         # ingest without pruning
    python run_ingest_batched.py --batch-size 2     # 2 parallel per batch
    python run_ingest_batched.py --max 5            # only first 5 conversations
"""
import json
import re
import time
import threading
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from argparse import ArgumentParser
from typing import List, Dict, Any, Optional, Tuple

API_BASE = "http://localhost:8000"
DATA_PATH = "../../../dataset/locomo10.json"

# Default batch sizes: [3, 3, 4]
DEFAULT_BATCHES = [3, 3, 4]

# Pipeline status poll interval (seconds)
POLL_INTERVAL = 30

# Session completion poll interval (seconds) — more frequent than batch-level
SESSION_POLL_INTERVAL = 10

# Thread-safe token refresh lock (RLock to avoid deadlock)
_token_lock = threading.RLock()  # Reentrant: same thread may acquire multiple times
_shared_token = None

# LOCOMO time format regex: "1:56 pm on 8 May, 2023"
LOCOMO_TIME_PATTERN = re.compile(
    r"(\d{1,2}):(\d{2})\s*(am|pm)\s+on\s+(\d{1,2})\s+(\w+),?\s*(\d{4})",
    re.IGNORECASE
)

# Month name mapping
MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12
}


def parse_locomo_datetime(time_str: str) -> Optional[datetime]:
    """
    Parse a LOCOMO-style time string.

    Supports two formats:
    1. LOCOMO raw: "1:56 pm on 8 May, 2023"
    2. ISO: "2023-10-15 14:30:00" or "2023-10-15T14:30:00"

    Args:
        time_str: Time string

    Returns:
        datetime in UTC, or None if parsing fails
    """
    if not time_str:
        return None
    
    time_str = time_str.strip()
    
    # Try ISO formats: "2023-10-15 14:30:00" or "2023-10-15T14:30:00"
    iso_patterns = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%dT%H:%M",
        "%Y-%m-%d",
    ]
    for pattern in iso_patterns:
        try:
            dt = datetime.strptime(time_str, pattern)
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    
    # Try LOCOMO raw format: "1:56 pm on 8 May, 2023"
    match = LOCOMO_TIME_PATTERN.search(time_str)
    if match:
        hour, minute, meridiem, day, month_name, year = match.groups()
        
        hour = int(hour)
        minute = int(minute)
        if meridiem.lower() == "pm" and hour != 12:
            hour += 12
        elif meridiem.lower() == "am" and hour == 12:
            hour = 0
        
        month = MONTH_MAP.get(month_name.lower())
        if month:
            try:
                return datetime(
                    year=int(year),
                    month=month,
                    day=int(day),
                    hour=hour,
                    minute=minute,
                    tzinfo=timezone.utc
                )
            except ValueError:
                pass
    
    return None


def datetime_to_iso(dt: Optional[datetime]) -> Optional[str]:
    """Convert datetime to ISO 8601 string."""
    if dt is None:
        return None
    return dt.isoformat()


def datetime_to_ms(dt: Optional[datetime]) -> Optional[int]:
    """Convert datetime to Unix ms timestamp."""
    if dt is None:
        return None
    return int(dt.timestamp() * 1000)


def format_date_for_message(dt: Optional[datetime]) -> str:
    """
    Format time as message prefix.

    Args:
        dt: datetime instance

    Returns:
        e.g. "[May 08, 2023, 1:56 PM]", or empty string on failure
    """
    if dt is None:
        return ""
    return dt.strftime("[%B %d, %Y, %I:%M %p]")


def api_call(method: str, endpoint: str, data: Any = None, token: str = None) -> Tuple[Any, int]:
    """Call API; returns (response_json, status_code)."""
    url = f"{API_BASE}{endpoint}"
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    
    body = json.dumps(data).encode() if data else None
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    
    try:
        with urllib.request.urlopen(req) as resp:
            return json.load(resp), resp.status
    except urllib.error.HTTPError as e:
        try:
            return json.load(e), e.code
        except Exception:
            return {"detail": str(e)}, e.code
    except urllib.error.URLError as e:
        return {"detail": f"Connection error: {e}"}, 0


def register_user() -> bool:
    """Register default user; returns whether it succeeded."""
    try:
        data = json.dumps({
            "email": "default_user@example.com",
            "password": "default_password"
        }).encode()
        req = urllib.request.Request(
            f"{API_BASE}/api/v1/auth/register",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req) as resp:
            return resp.status == 201
    except urllib.error.HTTPError as e:
        if e.code == 400:
            # User already exists — not an error
            return True
        return False
    except Exception:
        return False


def login(max_retries: int = 60, retry_delay: int = 5) -> str:
    """Log in for token; auto-registers if user does not exist."""
    login_data = "username=default_user@example.com&password=default_password"
    registered = False
    
    for i in range(max_retries):
        try:
            req = urllib.request.Request(
                f"{API_BASE}/api/v1/auth/login",
                data=login_data.encode(),
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            with urllib.request.urlopen(req) as resp:
                return json.load(resp)["access_token"]
        except urllib.error.HTTPError as e:
            # Login failed — try registering user
            if not registered:
                print("  User not found, attempting registration...")
                if register_user():
                    print("  ✓ User registered successfully")
                    registered = True
                    continue  # Retry login

            if i < max_retries - 1:
                print(f"  Login retry ({i+1}/{max_retries})...")
                time.sleep(retry_delay)
            else:
                raise RuntimeError(f"Login failed: {e.code}")
        except Exception as e:
            if i < max_retries - 1:
                print(f"  Connection retry ({i+1}/{max_retries}): {e}")
                time.sleep(retry_delay)
            else:
                raise


def get_active_pipelines(token: str) -> Tuple[List[Dict], bool]:
    """
    List all active pipelines.

    Returns:
        (pipelines, success): pipeline list and whether the call succeeded.
        Returns ([], False) if token is invalid.
    """
    result, status = api_call("GET", "/api/v1/pipeline/active", token=token)
    if status == 200:
        # API returns a list or {"pipelines": [...]}
        if isinstance(result, list):
            return result, True
        return result.get("pipelines", []), True
    if status in (401, 403):
        # Invalid token
        return [], False
    return [], True  # Other errors but not auth-related


def cancel_all_pipelines(token: str) -> int:
    """Cancel all active pipelines; returns count cancelled."""
    pipelines, _ = get_active_pipelines(token)
    cancelled = 0
    for p in pipelines:
        # API uses camelCase field names
        run_id = p.get("pipelineRunId")
        if run_id:
            result, status = api_call("DELETE", f"/api/v1/pipeline/active/{run_id}", token=token)
            if status == 200:
                cancelled += 1
    return cancelled


def prune_all(token: str) -> Tuple[Any, int]:
    """Prune database; cancels active pipelines as needed."""
    while True:
        result, status = api_call("POST", "/api/v1/prune/all", {"confirm": "DELETE_ALL_DATA"}, token)
        
        if status == 200:
            return result, status
        
        if status == 409 and "pipeline" in str(result):
            print(f"  Active pipelines present, cancelling...")
            cancelled = cancel_all_pipelines(token)
            print(f"  Cancelled {cancelled} pipeline(s)")
            time.sleep(2)
            continue
        
        return result, status


def format_single_session(
    conv: Dict,
    session_key: str,
    include_blip_caption: bool = True
) -> Tuple[str, Optional[datetime]]:
    """
    Format one session; add time prefix to each message.

    Args:
        conv: Conversation data dict
        session_key: Session key (e.g. "session_1")
        include_blip_caption: Whether to include image description (blip_caption)

    Returns:
        (formatted_content, session_datetime)
    """
    lines = []
    
    ts_str = conv.get(f"{session_key}_date_time", "")
    session_dt = parse_locomo_datetime(ts_str)
    
    # Build message time prefix
    time_prefix = format_date_for_message(session_dt)
    
    for msg in conv.get(session_key, []):
        speaker = msg.get("speaker", "Unknown")
        text = msg.get("text", "")
        
        # Build message line with optional time prefix
        if time_prefix:
            msg_line = f"{time_prefix} {speaker}: {text}"
        else:
            msg_line = f"{speaker}: {text}"
        
        lines.append(msg_line)
        
        # Append image description (blip_caption) as its own line
        if include_blip_caption:
            blip_caption = msg.get("blip_caption", "")
            if blip_caption:
                lines.append(f"  [Image shared by {speaker}: {blip_caption}]")
    
    return "\n".join(lines), session_dt


def format_conversation_with_sessions(
    conv: Dict, 
    include_blip_caption: bool = True
) -> Tuple[str, Optional[datetime]]:
    """
    Format conversation; add time prefix to each message.

    Time strategy:
    1. Parse each session timestamp to datetime
    2. Prefix each message (e.g. "[May 08, 2023, 1:56 PM]")
    3. Keep session boundary markers
    4. Return earliest session time as created_at anchor

    Args:
        conv: Conversation data dict
        include_blip_caption: Whether to include image description (blip_caption)

    Returns:
        (formatted_content, earliest_datetime)
    """
    lines = []
    earliest_dt: Optional[datetime] = None
    
    # Sort sessions by numeric suffix
    session_keys = sorted(
        [k for k in conv.keys() if k.startswith("session_") and not k.endswith("_date_time")],
        key=lambda x: int(x.replace("session_", ""))
    )
    
    for key in session_keys:
        ts_str = conv.get(f"{key}_date_time", "")
        session_dt = parse_locomo_datetime(ts_str)
        
        # Track earliest time
        if session_dt is not None:
            if earliest_dt is None or session_dt < earliest_dt:
                earliest_dt = session_dt
        
        # Session boundary marker (ISO time when available)
        if session_dt:
            iso_time = datetime_to_iso(session_dt)
            lines.append(f"\n=== {key.upper()}: {iso_time} ===\n")
        else:
            lines.append(f"\n=== {key.upper()}: {ts_str} ===\n")
        
        # Message time prefix
        time_prefix = format_date_for_message(session_dt)
        
        for msg in conv.get(key, []):
            speaker = msg.get("speaker", "Unknown")
            text = msg.get("text", "")
            
            # Build message line with optional time prefix
            if time_prefix:
                msg_line = f"{time_prefix} {speaker}: {text}"
            else:
                msg_line = f"{speaker}: {text}"
            
            lines.append(msg_line)
            
            # Append image description (blip_caption) as its own line
            if include_blip_caption:
                blip_caption = msg.get("blip_caption", "")
                if blip_caption:
                    # Clearer format for image content
                    lines.append(f"  [Image shared by {speaker}: {blip_caption}]")
    
    return "\n".join(lines), earliest_dt


def start_ingest(
    dataset_name: str, 
    content: str, 
    token: str,
    created_at: Optional[str] = None
) -> Tuple[bool, Dict]:
    """
    Start ingest for one dataset (background mode).

    Args:
        dataset_name: Dataset name
        content: Content to ingest
        token: Auth token
        created_at: Content timestamp (ISO 8601 string) for relative-time parsing

    Returns:
        (success, result_dict)

        result_dict may include:
        - dataset_id
        - dataset_name
        - status: "background_started" | "completed" | "memorize_failed"
        - add_run_id
        - memorize_run_id
    """
    payload = {
        "content": content,
        "dataset_name": dataset_name,
        "enable_content_routing": True,
        "content_type": "dialog",  # LOCOMO is dialog data; set explicitly
        "enable_episode_routing": False,
        "precise_mode": True,  # Zero-loss summarization — required to reproduce benchmark scores
        "run_in_background": True,
        # Multiple ingests per dataset (one per session): avoid incremental/cache treating later batches as already done
        "incremental_loading": False,
        "use_pipeline_cache": False,
    }
    
    # Optional created_at
    if created_at:
        payload["created_at"] = created_at
    
    result, status = api_call("POST", "/api/v1/ingest", payload, token)
    
    # Check start succeeded
    if status != 200:
        return False, result

    # Check response status
    ingest_status = result.get("status", "")
    if ingest_status == "memorize_failed":
        return False, result
    
    return True, result


def wait_for_batch_completion(
    dataset_names: List[str],
    token: str,
    poll_interval: int = POLL_INTERVAL,
    initial_wait: int = 5,
    refresh_token_fn=None,
) -> Tuple[Dict[str, str], str]:
    """
    Wait until all pipelines for a batch of datasets finish (no overall timeout).

    Safety:
    1. Initial wait so pipelines appear in the active list
    2. Double confirmation: dataset must be absent from active list twice
    3. Log pipeline details including add/memorize
    4. Auto-refresh token on expiry

    Args:
        refresh_token_fn: Callable with no args returning new token

    Returns:
        ({dataset_name: status}, current_token)
    """
    start_time = time.time()
    current_token = token
    
    # State: "waiting" -> "running" -> "confirming" -> "completed"
    # "confirming": last poll was absent from active list; need one more confirm
    dataset_status = {name: "waiting" for name in dataset_names}

    # Initial wait so pipelines register in DB
    print(f"  Initial wait {initial_wait}s for pipelines to register...")
    time.sleep(initial_wait)
    
    while True:
        # Active pipelines (and token validity)
        active_pipelines, auth_ok = get_active_pipelines(current_token)

        # Expired token — refresh
        if not auth_ok:
            if refresh_token_fn:
                print("  ⚠ Token expired, refreshing...")
                try:
                    current_token = refresh_token_fn()
                    print("  ✓ Token refreshed")
                    active_pipelines, auth_ok = get_active_pipelines(current_token)
                    if not auth_ok:
                        print("  ✗ Still unauthorized after refresh")
                        return dataset_status, current_token
                except Exception as e:
                    print(f"  ✗ Token refresh failed: {e}")
                    return dataset_status, current_token
            else:
                print("  ✗ Token expired and no refresh function")
                return dataset_status, current_token

        # dataset -> pipelines map
        # Note: API uses camelCase (datasetName, pipelineName, ...)
        dataset_pipelines: Dict[str, List[Dict]] = {}
        for p in active_pipelines:
            ds_name = p.get("datasetName", "")
            if ds_name:
                if ds_name not in dataset_pipelines:
                    dataset_pipelines[ds_name] = []
                dataset_pipelines[ds_name].append(p)
        
        # Update per-dataset state
        for name in dataset_names:
            if name in dataset_pipelines:
                # Still has active pipelines
                dataset_status[name] = "running"
            else:
                # Not in active list
                if dataset_status[name] == "waiting":
                    # Not in list yet — may not be registered; keep waiting
                    # After enough elapsed time, move to confirming
                    if time.time() - start_time > initial_wait * 2:
                        dataset_status[name] = "confirming"
                elif dataset_status[name] == "running":
                    # Was running, now absent — need second confirm
                    dataset_status[name] = "confirming"
                elif dataset_status[name] == "confirming":
                    # Two consecutive absences — done
                    dataset_status[name] = "completed"
                # "completed" stays completed

        # All done?
        running_count = sum(1 for s in dataset_status.values() if s in ("waiting", "running", "confirming"))
        if running_count == 0:
            return dataset_status, current_token
        
        # Progress line
        waiting = sum(1 for s in dataset_status.values() if s == "waiting")
        running = sum(1 for s in dataset_status.values() if s == "running")
        confirming = sum(1 for s in dataset_status.values() if s == "confirming")
        completed = sum(1 for s in dataset_status.values() if s == "completed")
        elapsed = int(time.time() - start_time)
        
        # Elapsed time string
        hours, remainder = divmod(elapsed, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours > 0:
            time_str = f"{hours}h{minutes}m"
        else:
            time_str = f"{minutes}m{seconds}s"
        
        # Active pipeline summary
        pipeline_summary = []
        for name in dataset_names:
            if name in dataset_pipelines:
                pipes = dataset_pipelines[name]
                pipe_names = [p.get("pipelineName", "?")[:3] for p in pipes]
                pipeline_summary.append(f"{name.split('_')[-1]}:{'+'.join(pipe_names)}")
        
        status_str = f"wait:{waiting} run:{running} confirm:{confirming} done:{completed}"
        active_str = ', '.join(pipeline_summary[:5]) if pipeline_summary else "none"
        print(f"  [{time_str}] {status_str} | active: {active_str}")
        
        time.sleep(poll_interval)


def wait_for_dataset_idle(
    dataset_name: str,
    token: str,
    poll_interval: int = SESSION_POLL_INTERVAL,
    initial_wait: float = 5.0,  # Longer initial wait so pipelines register
    max_wait: int = 900,  # 15 minutes for very large sessions
    refresh_token_fn=None,
) -> Tuple[bool, str]:
    """
    Wait until all pipelines for one dataset finish (session-level serial ingest).

    Design:
    1. Initial wait so pipelines register in DB
    2. Must see active pipeline (seen_active=True) before treating idle as done
    3. Multiple consecutive idle polls to avoid false positives

    Args:
        dataset_name: Dataset to wait on
        token: Auth token
        poll_interval: Poll interval (seconds)
        initial_wait: Initial wait (seconds) for registration
        max_wait: Max wait (seconds)
        refresh_token_fn: Token refresh callable

    Returns:
        (success, current_token)
    """
    global _shared_token
    start_time = time.time()
    current_token = token
    
    # Initial wait for pipeline registration
    time.sleep(initial_wait)

    # State
    seen_active = False  # Whether we ever saw an active pipeline for this dataset
    idle_count = 0
    required_idle_checks = 2  # Two consecutive idle polls (reliability vs speed)

    # First check after initial_wait
    # Use latest shared token for consistency
    with _token_lock:
        check_token = _shared_token if _shared_token else token
    first_check_pipelines, first_auth_ok = get_active_pipelines(check_token)
    if first_auth_ok:
        if any(p.get("datasetName") == dataset_name for p in first_check_pipelines):
            seen_active = True
    
    while time.time() - start_time < max_wait:
        # Active pipelines with thread-safe token
        with _token_lock:
            if _shared_token:
                current_token = _shared_token
        
        active_pipelines, auth_ok = get_active_pipelines(current_token)
        
        # Token expiry (thread-safe)
        if not auth_ok:
            if refresh_token_fn:
                with _token_lock:
                    try:
                        new_token = refresh_token_fn()
                        _shared_token = new_token
                        current_token = new_token
                        active_pipelines, auth_ok = get_active_pipelines(current_token)
                        if not auth_ok:
                            return False, current_token
                    except Exception:
                        return False, current_token
            else:
                return False, current_token
        
        # Any active pipeline for this dataset?
        dataset_active = any(
            p.get("datasetName") == dataset_name 
            for p in active_pipelines
        )
        
        if dataset_active:
            seen_active = True
            idle_count = 0  # Reset idle streak
        else:
            if seen_active:
                # Was active, now idle — count toward completion
                idle_count += 1
                if idle_count >= required_idle_checks:
                    return True, current_token
            else:
                # Never saw active
                elapsed = time.time() - start_time
                if elapsed > initial_wait * 2:
                    # Long wait with no active — fast finish or ingest failed
                    idle_count += 1
                    if idle_count >= required_idle_checks:
                        return True, current_token
        
        time.sleep(poll_interval)
    
    # Timeout: return True to avoid blocking callers (very long session)
    return True, current_token


def ingest_dataset_serial(
    dataset: Dict,
    token: str,
    refresh_token_fn=None,
    verbose: bool = False,
) -> Tuple[str, str, int, int, str]:
    """
    Ingest all sessions of one dataset serially.

    Design:
    1. Start next session only after previous finishes (including indexing)
    2. So episode routing can see prior session content
    3. Thread-safe token handling

    Args:
        dataset: Dataset dict with name, sessions, etc.
        token: Auth token
        refresh_token_fn: Token refresh callable
        verbose: Verbose logging

    Returns:
        (dataset_name, status, success_count, total_count, current_token)
    """
    global _shared_token
    
    dataset_name = dataset['name']
    sessions = dataset.get('sessions', [])
    total = len(sessions)
    success_count = 0
    failed_sessions = []
    
    # Shared token (thread-safe)
    with _token_lock:
        if _shared_token:
            current_token = _shared_token
        else:
            current_token = token
            _shared_token = token
    
    for i, sess in enumerate(sessions):
        created_at = sess.get('created_at')
        session_key = sess.get('session_key', f'session_{i}')
        
        # Latest token
        with _token_lock:
            if _shared_token:
                current_token = _shared_token
        
        # 1. Start ingest
        success, result = start_ingest(
            dataset_name,
            sess['content'],
            current_token,
            created_at=created_at
        )
        
        if not success:
            # Token expiry (thread-safe)
            if isinstance(result, dict) and result.get('detail') == 'Unauthorized':
                if refresh_token_fn:
                    with _token_lock:
                        try:
                            new_token = refresh_token_fn()
                            _shared_token = new_token
                            current_token = new_token
                        except Exception:
                            pass
                    
                    # Retry
                    success, result = start_ingest(
                        dataset_name, sess['content'], current_token, created_at=created_at
                    )
            
            if not success:
                failed_sessions.append(session_key)
                if verbose:
                    print(f"      [{dataset_name}] {session_key} ingest failed: {result}")
                # Skip wait on failure; move to next session
                continue

        # 2. Wait for this session's pipelines (critical)
        completed, current_token = wait_for_dataset_idle(
            dataset_name,
            current_token,
            refresh_token_fn=refresh_token_fn
        )
        
        # Update shared token
        with _token_lock:
            _shared_token = current_token
        
        if completed:
            success_count += 1
            # Progress: first, last, every 5th
            if verbose or (i + 1) % 5 == 0 or i == 0 or i == total - 1:
                print(f"      [{dataset_name}] {i+1}/{total} {session_key} ✓")
        else:
            # Auth failure (token refresh) — not counted success; continue
            # Note: wait_for_dataset_idle returns True on timeout (does not block downstream)
            if verbose:
                print(f"      [{dataset_name}] {session_key} wait failed (auth)")
            failed_sessions.append(f"{session_key}(auth_failed)")
    
    # Final status
    if success_count == total:
        status = "completed"
    elif success_count > 0:
        status = "partial"
    else:
        status = "failed"
    
    # Failed session list if verbose
    if failed_sessions and verbose:
        print(f"      [{dataset_name}] failed sessions: {', '.join(failed_sessions)}")
    
    return dataset_name, status, success_count, total, current_token


def load_conversations(data_path: str, max_count: Optional[int] = None) -> List[Dict]:
    """
    Load conversation data.

    Supports:
    - locomo10_rag.json: dict {"0": {...}, "1": {...}}
    - locomo10.json: list [{...}, {...}]
    """
    with open(data_path) as f:
        data = json.load(f)

    # Dict shape (locomo10_rag.json)
    if isinstance(data, dict):
        # Sort by numeric key
        sorted_keys = sorted(data.keys(), key=lambda x: int(x))
        data = [data[k] for k in sorted_keys]
    
    if max_count:
        data = data[:max_count]
    
    return data


def prepare_datasets(conversations: List[Dict]) -> List[Dict]:
    """
    Build dataset list — one entry per conversation with per-session records.

    Formats:
    - locomo10_rag.json: conversation is message list; sessions split on timestamp changes
    - locomo10.json: nested dict with session_X and session_X_date_time

    Time:
    - Each session ingested separately
    - Each session uses its own timestamp as created_at
    - All sessions of one conversation share one dataset

    Returns:
        [{name, sessions: [...], speakers, total_messages, index}, ...]
    """
    datasets = []
    
    for idx, item in enumerate(conversations):
        conv = item["conversation"]
        
        # List (locomo10_rag) vs dict (locomo10)
        if isinstance(conv, list):
            # locomo10_rag: message list
            sessions_info = _prepare_sessions_from_list(conv)

            # Speakers from messages
            speakers_set = set()
            for msg in conv[:20]:
                speakers_set.add(msg.get("speaker", "Unknown"))
            speakers_list = sorted(speakers_set)
            speaker_a = speakers_list[0] if len(speakers_list) > 0 else "A"
            speaker_b = speakers_list[1] if len(speakers_list) > 1 else "B"
        else:
            # locomo10: nested dict
            speaker_a = conv.get("speaker_a", "A")
            speaker_b = conv.get("speaker_b", "B")
            sessions_info = _prepare_sessions_from_dict(conv)
        
        # Shared dataset name (match search script)
        dataset_name = f"locomo_benchmark_{speaker_a}_{speaker_b}_{idx}"

        # One entry with all sessions for this conversation
        if sessions_info:
            datasets.append({
                "name": dataset_name,
                "sessions": sessions_info,
                "speakers": f"{speaker_a} <-> {speaker_b}",
                "total_messages": sum(s["message_count"] for s in sessions_info),
                "index": idx,
            })
    
    return datasets


def _prepare_sessions_from_list(messages: List[Dict]) -> List[Dict]:
    """
    Build sessions from message list (locomo10_rag.json).

    Split sessions when timestamp changes.
    """
    sessions_info = []

    # Split messages into sessions by timestamp
    current_ts = None
    current_session_msgs = []
    session_idx = 1
    
    for msg in messages:
        ts = msg.get("timestamp", "")
        
        if ts != current_ts and current_session_msgs:
            # New session — flush current
            session_info = _format_session_from_msgs(
                current_session_msgs, 
                current_ts, 
                f"session_{session_idx}"
            )
            sessions_info.append(session_info)
            session_idx += 1
            current_session_msgs = []
        
        current_ts = ts
        current_session_msgs.append(msg)
    
    # Last session
    if current_session_msgs:
        session_info = _format_session_from_msgs(
            current_session_msgs, 
            current_ts, 
            f"session_{session_idx}"
        )
        sessions_info.append(session_info)
    
    return sessions_info


def _format_session_from_msgs(
    messages: List[Dict], 
    timestamp_str: str, 
    session_key: str
) -> Dict:
    """Format message list into one session record."""
    session_dt = parse_locomo_datetime(timestamp_str)
    time_prefix = format_date_for_message(session_dt)
    
    lines = []
    for msg in messages:
        speaker = msg.get("speaker", "Unknown")
        text = msg.get("text", "")
        
        if time_prefix:
            msg_line = f"{time_prefix} {speaker}: {text}"
        else:
            msg_line = f"{speaker}: {text}"
        
        lines.append(msg_line)
    
    content = "\n".join(lines)
    created_at_iso = datetime_to_iso(session_dt)
    
    return {
        "content": content,
        "message_count": len(messages),
        "created_at": created_at_iso,
        "created_at_display": timestamp_str,
        "session_key": session_key,
    }


def _prepare_sessions_from_dict(conv: Dict) -> List[Dict]:
    """
    Build sessions from nested dict (locomo10.json).
    """
    sessions_info = []

    # Session keys, sorted
    session_keys = sorted(
        [k for k in conv.keys() if k.startswith("session_") and not k.endswith("_date_time")],
        key=lambda x: int(x.replace("session_", ""))
    )
    
    for session_key in session_keys:
        content, session_dt = format_single_session(conv, session_key)
        
        if not content.strip():
            continue
        
        msg_count = len(conv.get(session_key, []))
        created_at_iso = datetime_to_iso(session_dt)
        ts_str = conv.get(f"{session_key}_date_time", "")
        
        sessions_info.append({
            "content": content,
            "message_count": msg_count,
            "created_at": created_at_iso,
            "created_at_display": ts_str,
            "session_key": session_key,
        })
    
    return sessions_info


def create_batches(datasets: List[Dict], batch_sizes: List[int]) -> List[List[Dict]]:
    """
    Chunk datasets according to batch_sizes.

    e.g. batch_sizes=[3,3,4] splits 10 datasets into 3 batches.
    """
    batches = []
    offset = 0
    
    for size in batch_sizes:
        batch = datasets[offset:offset + size]
        if batch:
            batches.append(batch)
        offset += size
    
    # Trailing slice as final batch
    remaining = datasets[offset:]
    if remaining:
        batches.append(remaining)
    
    return batches


def main():
    parser = ArgumentParser(description="LOCOMO batched concurrent ingest script")
    parser.add_argument("--no-prune", action="store_true", help="Skip database prune")
    parser.add_argument("--force", action="store_true", help="Skip confirmation; auto-cancel active pipelines")
    parser.add_argument("--max", type=int, help="Process at most N conversations")
    parser.add_argument("--data", default=DATA_PATH, help="Path to data file")
    parser.add_argument("--batch-size", type=int, help="Parallel count per batch (overrides default 3-3-4)")
    parser.add_argument("--poll", type=int, default=POLL_INTERVAL, help="Status poll interval in seconds")
    args = parser.parse_args()
    
    print("=" * 60)
    print("LOCOMO batched concurrent ingest")
    print("=" * 60)

    # === Step 1: Login ===
    print("\n[Step 1] Login")
    token = login()
    print("  ✓ Logged in")

    # === Step 2: Check / clear active pipelines ===
    print("\n[Step 2] Check active pipelines")
    active, _ = get_active_pipelines(token)
    if active:
        print(f"  ⚠ Found {len(active)} active pipeline(s)")
        if args.force or not args.no_prune:
            print("  Cancelling...")
            cancelled = cancel_all_pipelines(token)
            print(f"  ✓ Cancelled {cancelled}")
            # Wait for pipeline state to update
            time.sleep(3)
        else:
            print("  Warning: --no-prune mode, skipping cancel")
            print("  This may conflict with existing pipelines!")
            print("  Use --force to auto-cancel")
            try:
                response = input("  Continue? [y/N]: ")
                if response.lower() != 'y':
                    print("  Aborted")
                    return
            except EOFError:
                print("\n  Cannot read input; use --force to skip confirmation")
                return
    else:
        print("  ✓ No active pipelines")

    # === Step 3: Prune DB (if requested) ===
    if not args.no_prune:
        print("\n[Step 3] Prune database")
        result, status = prune_all(token)
        if status != 200:
            print(f"  ✗ Prune failed: {result}")
            return
        print("  ✓ Database pruned")

        # Re-login
        print("  Re-logging in...")
        time.sleep(3)
        token = login()
        print("  ✓ Re-login successful")
    else:
        print("\n[Step 3] Skipping prune (--no-prune)")

    # === Step 4: Load data ===
    print(f"\n[Step 4] Load data: {args.data}")
    conversations = load_conversations(args.data, args.max)
    datasets = prepare_datasets(conversations)
    
    # Count conversations and sessions
    total_sessions = sum(len(ds.get('sessions', [])) for ds in datasets)
    print(f"  ✓ {len(datasets)} conversation(s), {total_sessions} session(s)")

    # Per-conversation session list
    for ds in datasets:
        print(f"    [{ds['index']}] {ds['speakers']}:")
        for sess in ds.get('sessions', []):
            created_at_str = sess.get('created_at_display', sess.get('created_at', 'N/A'))
            print(f"        {sess['session_key']}: {sess['message_count']} messages @ {created_at_str}")
    
    # === Step 5: Build batches ===
    print("\n[Step 5] Batch plan")
    if args.batch_size:
        # Even-sized batches
        batch_sizes = []
        remaining = len(datasets)
        while remaining > 0:
            size = min(args.batch_size, remaining)
            batch_sizes.append(size)
            remaining -= size
    else:
        # Default 3-3-4
        batch_sizes = DEFAULT_BATCHES[:len(datasets)]
        # Pad final batch
        total_in_batches = sum(batch_sizes)
        if total_in_batches < len(datasets):
            batch_sizes.append(len(datasets) - total_in_batches)
    
    batches = create_batches(datasets, batch_sizes)
    print(f"  Batch sizes: {[len(b) for b in batches]} ({len(datasets)} conversation(s) total)")
    for i, batch in enumerate(batches):
        # Conversation indices in batch
        conv_info = [f"conv{ds['index']}" for ds in batch]
        print(f"    Batch {i+1}: {conv_info}")
    
    # === Step 6: Batched ingest (serial sessions per dataset, parallel datasets) ===
    print("\n[Step 6] Start batched ingest")
    print("  Strategy: sessions serial within each dataset (wait for indexing)")
    print("            datasets processed in parallel across workers")
    total_start = time.time()
    
    all_results = []
    
    # Shared token (thread-safe)
    global _shared_token
    _shared_token = token

    # Token refresh (thread-safe)
    def refresh_token():
        global _shared_token
        with _token_lock:
            new_token = login(max_retries=3, retry_delay=2)
            _shared_token = new_token
            return new_token
    
    for batch_idx, batch in enumerate(batches):
        batch_num = batch_idx + 1
        print(f"\n{'='*40}")
        print(f"Batch {batch_num}/{len(batches)}: {len(batch)} conversation(s) in parallel")
        print(f"{'='*40}")

        # Refresh token before each batch (after first)
        if batch_idx > 0:
            print("  Refreshing token...")
            try:
                token = refresh_token()
                print("  ✓ Token refreshed")
            except Exception as e:
                print(f"  ✗ Token refresh failed: {e}")
                print("  Continuing with existing token...")
        
        batch_start = time.time()
        
        # Batch summary
        for ds in batch:
            sessions = ds.get('sessions', [])
            print(f"  → Conv {ds['index']}: {ds['name']} ({len(sessions)} sessions)")
        
        # 6.1 ThreadPoolExecutor: parallel datasets
        # Each dataset serializes sessions (wait for indexing per session)
        print(f"\n  Starting {len(batch)} parallel dataset worker(s)...")
        
        batch_results = []
        
        with ThreadPoolExecutor(max_workers=len(batch)) as executor:
            # One future per dataset
            future_to_ds = {
                executor.submit(
                    ingest_dataset_serial,
                    ds,
                    token,
                    refresh_token_fn=refresh_token,
                    verbose=False,  # Less noisy output
                ): ds
                for ds in batch
            }
            
            # Collect results
            for future in as_completed(future_to_ds):
                ds = future_to_ds[future]
                try:
                    ds_name, status, success, total, _ = future.result()
                    batch_results.append({
                        "name": ds_name,
                        "status": status,
                        "success": success,
                        "total": total,
                    })
                    symbol = "✓" if status == "completed" else "⚠"
                    print(f"    {symbol} {ds_name}: {success}/{total} sessions")
                except Exception as e:
                    batch_results.append({
                        "name": ds['name'],
                        "status": "error",
                        "error": str(e),
                    })
                    print(f"    ✗ {ds['name']}: error - {e}")
        
        batch_duration = time.time() - batch_start
        
        # 6.2 Record results
        for r in batch_results:
            all_results.append({"name": r['name'], "status": r['status']})
        
        # 6.3 Batch summary
        completed = sum(1 for r in batch_results if r['status'] == 'completed')

        print(f"\n  Batch {batch_num} results:")
        print(f"    Completed: {completed}/{len(batch)}")
        print(f"    Duration: {batch_duration/60:.1f} min")
        
        for r in batch_results:
            symbol = "✓" if r['status'] == 'completed' else "✗"
            extra = f" ({r.get('success', '?')}/{r.get('total', '?')})" if 'success' in r else ""
            print(f"      {symbol} {r['name']}: {r['status']}{extra}")
    
    # === Step 7: Summary ===
    total_duration = time.time() - total_start

    print("\n" + "=" * 60)
    print("Ingest summary")
    print("=" * 60)
    
    completed = sum(1 for r in all_results if r['status'] == 'completed')
    failed = len(all_results) - completed
    
    # Total elapsed string
    hours, remainder = divmod(int(total_duration), 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours > 0:
        time_str = f"{hours}h {minutes}m"
    else:
        time_str = f"{minutes}m {seconds}s"

    total_sessions = sum(len(ds.get('sessions', [])) for ds in datasets)
    print(f"  Total conversations: {len(datasets)}")
    print(f"  Total sessions: {total_sessions}")
    print(f"  Succeeded: {completed}")
    print(f"  Failed: {failed}")
    print(f"  Total time: {time_str}")

    if failed > 0:
        print("\n  Failure details:")
        for r in all_results:
            if r['status'] != 'completed':
                print(f"    - {r['name']}: {r['status']}")
    
    # Final pipeline check
    print("\n  Final pipeline status:")
    final_active, _ = get_active_pipelines(token)
    if final_active:
        print(f"    ⚠ Still {len(final_active)} active pipeline(s)")
        for p in final_active[:5]:
            print(f"      - {p.get('datasetName')}: {p.get('pipelineName')}")
    else:
        print("    ✓ No active pipelines")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
