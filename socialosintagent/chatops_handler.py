"""
Telegram ChatOps entrypoint (Phase 1): drive SocialOSINTAgent from a bot.

Environment:
    TELEGRAM_BOT_TOKEN — from `@BotFather`

Run:
    python -m socialosintagent.chatops_handler

Commands:
    /analyze <platform>/<username> — default OSINT query, reply with Markdown
    /help — usage
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import uuid
from pathlib import Path
from typing import List, Optional, Tuple
from datetime import datetime, timezone
import re

from aiogram import Bot, Dispatcher
from aiogram.filters import Command, CommandStart
from aiogram.types import Message
from dotenv import load_dotenv

from socialosintagent.analyzer import SocialOSINTAgent
from socialosintagent.cache import CacheManager
from socialosintagent.client_manager import ClientManager
from socialosintagent.llm import LLMAnalyzer
from socialosintagent.platforms import FETCHERS
from socialosintagent.utils import sanitize_username
from socialosintagent.session_manager import SessionManager

logger = logging.getLogger("SocialOSINTAgent.chatops")

# Telegram hard limit per message (leave margin for formatting)
TELEGRAM_MAX_LEN = 4000

DEFAULT_ANALYSIS_QUERY = (
    "Provide a structured OSINT analysis of this user: summarize public persona, "
    "recurring topics, communication style, and any notable patterns visible in "
    "the collected posts and profile. Use clear Markdown headings."
)


def chunk_telegram_text(text: str, max_len: int = TELEGRAM_MAX_LEN) -> List[str]:
    """Split long text into Telegram-sized chunks, preferring paragraph breaks."""
    if len(text) <= max_len:
        return [text]
    chunks: List[str] = []
    remaining = text
    while remaining:
        if len(remaining) <= max_len:
            chunks.append(remaining)
            break
        window = remaining[:max_len]
        split_at = window.rfind("\n\n")
        if split_at < max_len // 2:
            split_at = window.rfind("\n")
        if split_at < max_len // 2:
            split_at = max_len
        chunks.append(remaining[:split_at].rstrip())
        remaining = remaining[split_at:].lstrip()
    return chunks


def parse_analyze_command(text: str) -> Optional[Tuple[str, str]]:
    """
    Parse ``/analyze platform/username`` into (platform, username).

    Mastodon-style handles (user@host) are supported: only the first ``/``
    splits platform from the rest.
    """
    parts = (text or "").split(maxsplit=1)
    if len(parts) < 2:
        return None
    arg = parts[1].strip()
    if "/" not in arg:
        return None
    platform_raw, user_raw = arg.split("/", 1)
    platform = platform_raw.lower().strip()
    if not platform or not user_raw.strip():
        return None
    username = sanitize_username(user_raw.strip())
    if not username:
        return None
    return platform, username


def _strip_outer_quotes(text: str) -> str:
    t = (text or "").strip()
    if len(t) >= 2 and ((t[0] == t[-1] == '"') or (t[0] == t[-1] == "'")):
        return t[1:-1].strip()
    return t


def parse_monitor_command(text: str) -> Optional[Tuple[str, str, List[str], str]]:
    """
    Parse `/monitor <platform>/<username> for keywords "crypto, wallet"`.

    Returns:
        (platform, username, keywords, condition_string)
    """
    raw = (text or "").strip()
    if not raw.startswith("/monitor"):
        return None

    parts = raw.split(maxsplit=3)
    if len(parts) < 4:
        return None

    # Expected: /monitor <platform>/<username> for keywords <keywords...>
    arg_platform_user = parts[1]
    if "/" not in arg_platform_user:
        return None
    platform_raw, user_raw = arg_platform_user.split("/", 1)
    platform = platform_raw.lower().strip()
    username = sanitize_username(user_raw.strip())
    if not platform or not username:
        return None

    # `parts[3]` starts with: `keywords ...`
    remaining = parts[3].strip()
    if not remaining.lower().startswith("keywords"):
        return None

    keywords_raw = remaining[len("keywords") :].strip()
    keywords_raw = _strip_outer_quotes(keywords_raw)

    keywords = [k.strip() for k in keywords_raw.split(",") if k.strip()]
    if not keywords:
        return None

    condition = f"keywords: {', '.join(keywords)}"
    return platform, username, keywords, condition


def parse_monitor_discord_command(
    text: str,
) -> Optional[Tuple[str, str, List[str], str, str]]:
    """
    Parse:
      /monitor_discord <platform>/<username> for keywords "crypto, wallet" webhook "https://discord.../..."

    Returns:
      (platform, username, keywords, condition_string, webhook_url)
    """
    raw = (text or "").strip()
    if not raw.lower().startswith("/monitor_discord"):
        return None

    # Regex is resilient to whitespace and quoted strings.
    # - platform/username cannot contain whitespace or additional slashes
    # - keywords can be a quoted string containing commas/spaces
    # - webhook can be quoted or unquoted
    pat = re.compile(
        r"^/monitor_discord\s+"
        r"(?P<platform>[^/\s]+)/(?P<username>[^\s/]+)\s+"
        r"for\s+keywords\s+(?P<keywords>\".*?\"|'.*?'|[^ ]+)\s+"
        r"webhook\s+(?P<webhook>\".*?\"|'.*?'|https?://\S+)\s*$",
        re.IGNORECASE,
    )
    m = pat.match(raw)
    if not m:
        return None

    platform = (m.group("platform") or "").lower().strip()
    username = sanitize_username((m.group("username") or "").strip())
    if not platform or not username:
        return None

    keywords_raw = _strip_outer_quotes(m.group("keywords") or "")
    keywords = [k.strip() for k in keywords_raw.split(",") if k.strip()]
    if not keywords:
        return None

    webhook_url = _strip_outer_quotes((m.group("webhook") or "").strip())
    if not webhook_url.startswith("http"):
        return None

    condition = f"keywords: {', '.join(keywords)}"
    return platform, username, keywords, condition, webhook_url


def build_agent() -> SocialOSINTAgent:
    args = argparse.Namespace(
        offline=False,
        no_auto_save=True,
        format="markdown",
        unsafe_allow_external_media=False,
    )
    base_dir = Path("data")
    cache_manager = CacheManager(base_dir, args.offline)
    llm_analyzer = LLMAnalyzer(args.offline)
    client_manager = ClientManager(args.offline)
    return SocialOSINTAgent(args, cache_manager, llm_analyzer, client_manager)


def run_setup_logging() -> None:
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    log_file = logs_dir / "chatops.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stderr),
        ],
    )
    logging.getLogger("SocialOSINTAgent").setLevel(logging.INFO)


async def handle_start(message: Message) -> None:
    await message.answer(
        "OSINT ChatOps bot.\n\n"
        "Use:\n"
        "`/analyze <platform>/<username>`\n"
        "Example: `/analyze twitter/nasa`\n\n"
        "`/monitor <platform>/<username> for keywords \"crypto, wallet\"`\n"
        "Example: `/monitor bluesky/nasa for keywords \"crypto, wallet\"`\n\n"
        "`/monitor_discord <platform>/<username> for keywords \"crypto, wallet\" webhook \"https://discord.com/api/webhooks/...\"`\n\n"
        "Same engine as the CLI; ensure LLM and platform credentials are in `.env`."
    )


async def handle_help(message: Message) -> None:
    await handle_start(message)


async def handle_analyze(message: Message, agent: SocialOSINTAgent) -> None:
    parsed = parse_analyze_command(message.text or "")
    if not parsed:
        await message.answer(
            "Usage: `/analyze <platform>/<username>`\n"
            "Example: `/analyze twitter/username`",
        )
        return

    platform, username = parsed

    if platform not in FETCHERS:
        await message.answer(
            f"Unknown platform `{platform}`. "
            f"Known: {', '.join(sorted(FETCHERS.keys()))}",
        )
        return

    available = agent.client_manager.get_available_platforms(check_creds=True)
    if platform not in available:
        await message.answer(
            f"Platform `{platform}` is not configured (missing credentials). "
            f"Configured: {', '.join(available) or 'none'}",
        )
        return

    status_msg = await message.answer(
        f"Running analysis for `{platform}/{username}`… (this may take a minute)",
    )

    def run_analysis():
        return agent.analyze(
            {platform: [username]},
            DEFAULT_ANALYSIS_QUERY,
            force_refresh=False,
            fetch_options={"default_count": 50},
        )

    try:
        result = await asyncio.to_thread(run_analysis)
    except Exception as e:
        logger.exception("analyze failed: %s", e)
        await status_msg.edit_text(f"Analysis crashed: `{type(e).__name__}: {e}`")
        return

    if result.get("error"):
        report = result.get("report", "Unknown error")
        await status_msg.edit_text(f"Analysis failed:\n\n{report[: TELEGRAM_MAX_LEN - 100]}")
        return

    report = result.get("report") or ""
    try:
        await status_msg.delete()
    except Exception:
        pass

    chunks = chunk_telegram_text(report)
    for i, chunk in enumerate(chunks):
        prefix = f"(part {i + 1}/{len(chunks)})\n\n" if len(chunks) > 1 else ""
        await message.answer(prefix + chunk)


async def handle_monitor(message: Message, agent: SocialOSINTAgent) -> None:
    parsed = parse_monitor_command(message.text or "")
    if not parsed:
        await message.answer(
            "Usage:\n"
            "`/monitor <platform>/<username> for keywords \"crypto, wallet\"`\n"
            "Example: `/monitor bluesky/username for keywords \"crypto, wallet\"`",
        )
        return

    platform, username, keywords, condition = parsed

    if platform not in FETCHERS:
        await message.answer(f"Unknown platform `{platform}`. Known: {', '.join(sorted(FETCHERS.keys()))}")
        return

    available = agent.client_manager.get_available_platforms(check_creds=True)
    if platform not in available:
        await message.answer(
            f"Platform `{platform}` is not configured (missing credentials). "
            f"Configured: {', '.join(available) or 'none'}",
        )
        return

    session_manager = SessionManager(Path("data"))
    chat_id = int(message.chat.id)
    target = f"{platform}/{username}"
    now_iso = datetime.now(timezone.utc).isoformat()

    # Best-effort de-duplication: if the exact same rule already exists, do not create a new one.
    try:
        for path in session_manager.sessions_dir.glob("*.json"):
            session = session_manager.load(path.stem)
            if not session or not session.monitoring_rules:
                continue
            for rule in session.monitoring_rules:
                if (
                    rule.get("alert_channel") == chat_id
                    and rule.get("target") == target
                    and rule.get("condition") == condition
                ):
                    await message.answer(f"Already monitoring `{target}` for `{condition}`.")
                    return
    except Exception:
        # De-dupe is best-effort; proceed to create a rule even if scanning fails.
        logger.exception("De-duplication scan failed; proceeding to create a new rule.")

    session = session_manager.create(
        name=f"Monitor: {target}",
        platforms={platform: [username]},
        fetch_options={"default_count": 50, "targets": {}},
    )

    rule = {
        "rule_id": uuid.uuid4().hex[:8],
        "session_id": session.session_id,
        "target": target,
        "condition": condition,
        "alert_type": "telegram",
        "alert_channel": chat_id,
        "enabled": True,
        "created_at": now_iso,
        # Start from "now" so we only alert on posts newer than the rule registration time.
        "last_seen_post_created_at": now_iso,
    }
    session.monitoring_rules.append(rule)
    session_manager.save(session)

    await message.answer(
        "Monitoring enabled.\n"
        f"Target: `{target}`\n"
        f"Keywords: `{', '.join(keywords)}`\n"
        "I will check for new posts in the background and only alert you on matches.",
    )


async def handle_monitor_discord(message: Message, agent: SocialOSINTAgent) -> None:
    parsed = parse_monitor_discord_command(message.text or "")
    if not parsed:
        await message.answer(
            "Usage:\n"
            "`/monitor_discord <platform>/<username> for keywords \"crypto, wallet\" webhook \"https://discord.com/api/webhooks/...\"`"
        )
        return

    platform, username, keywords, condition, webhook_url = parsed

    if platform not in FETCHERS:
        await message.answer(f"Unknown platform `{platform}`. Known: {', '.join(sorted(FETCHERS.keys()))}")
        return

    available = agent.client_manager.get_available_platforms(check_creds=True)
    if platform not in available:
        await message.answer(
            f"Platform `{platform}` is not configured (missing credentials). "
            f"Configured: {', '.join(available) or 'none'}",
        )
        return

    session_manager = SessionManager(Path("data"))
    target = f"{platform}/{username}"
    now_iso = datetime.now(timezone.utc).isoformat()

    # Best-effort de-duplication: avoid creating duplicate identical rules.
    try:
        for path in session_manager.sessions_dir.glob("*.json"):
            session = session_manager.load(path.stem)
            if not session or not session.monitoring_rules:
                continue
            for rule in session.monitoring_rules:
                if (
                    rule.get("alert_type") == "discord"
                    and rule.get("alert_channel") == webhook_url
                    and rule.get("target") == target
                    and rule.get("condition") == condition
                ):
                    await message.answer(f"Already monitoring `{target}` for `{condition}` on this webhook.")
                    return
    except Exception:
        logger.exception("De-duplication scan failed; proceeding to create a new Discord rule.")

    session = session_manager.create(
        name=f"Discord Monitor: {target}",
        platforms={platform: [username]},
        fetch_options={"default_count": 50, "targets": {}},
    )

    rule = {
        "rule_id": uuid.uuid4().hex[:8],
        "session_id": session.session_id,
        "target": target,
        "condition": condition,
        "alert_type": "discord",
        "alert_channel": webhook_url,
        "enabled": True,
        "created_at": now_iso,
        "last_seen_post_created_at": now_iso,
    }
    session.monitoring_rules.append(rule)
    session_manager.save(session)

    await message.answer(
        "Discord monitoring enabled.\n"
        f"Target: `{target}`\n"
        f"Keywords: `{', '.join(keywords)}`\n"
        "I will check for new posts in the background and send alerts to your webhook only on matches.",
    )


def _register_handlers(dp: Dispatcher, agent: SocialOSINTAgent) -> None:
    dp.message.register(handle_start, CommandStart())
    dp.message.register(handle_help, Command("help"))
    dp.message.register(
        lambda m: handle_analyze(m, agent),
        Command("analyze"),
    )
    dp.message.register(
        lambda m: handle_monitor(m, agent),
        Command("monitor"),
    )
    dp.message.register(
        lambda m: handle_monitor_discord(m, agent),
        Command("monitor_discord"),
    )


async def main_async() -> None:
    load_dotenv()
    run_setup_logging()

    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        logger.error("TELEGRAM_BOT_TOKEN is not set.")
        sys.exit(1)

    try:
        agent = build_agent()
    except RuntimeError as e:
        logger.error("%s", e)
        sys.exit(1)

    bot = Bot(token=token)
    dp = Dispatcher()
    _register_handlers(dp, agent)
    logger.info("Starting Telegram polling…")
    await dp.start_polling(bot)


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
