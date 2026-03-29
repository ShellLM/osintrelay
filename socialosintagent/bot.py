"""
Telegram ChatOps daemon entrypoint (Phases 3–4).

Runs:
  - aiogram polling for commands (/analyze, /monitor, /help)
  - the background monitoring watcher loop
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from contextlib import suppress
from pathlib import Path

from aiogram import Bot, Dispatcher
from dotenv import load_dotenv

from .chatops_handler import build_agent, run_setup_logging, _register_handlers
from .session_manager import SessionManager
from .watcher import run_watcher_forever

logger = logging.getLogger("SocialOSINTAgent.bot")


async def main_async() -> None:
    load_dotenv()
    run_setup_logging()

    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        logger.error("TELEGRAM_BOT_TOKEN is not set.")
        sys.exit(1)

    agent = build_agent()
    bot = Bot(token=token)
    dp = Dispatcher()
    _register_handlers(dp, agent)

    session_manager = SessionManager(Path("data"))
    watcher_task = asyncio.create_task(
        run_watcher_forever(
            agent=agent,
            session_manager=session_manager,
            bot=bot,
        )
    )

    try:
        logger.info("Starting Telegram polling + background watcher.")
        await dp.start_polling(bot)
    finally:
        watcher_task.cancel()
        with suppress(asyncio.CancelledError):
            await watcher_task


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()

