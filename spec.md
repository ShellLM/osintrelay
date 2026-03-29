# Product Specification: OWASP Secure ChatOps OSINT Agent

# Product Specification: OWASP OSINT Relay

## 1. Vision & Context
**Context:** We are migrating our existing, highly-secure OSINT scraper ("OWASP Social OSINT Agent") into a continuous, always-on ChatOps agent named **OSINT Relay**. 
**Motivation:** Modern security teams live in Slack, Discord, and Telegram. Instead of running a CLI tool, users need a secure background agent that they can message to initiate investigations, and which will continuously monitor targets and *relay* intelligence back to their secure chat environment.
**Security Posture:** Unlike generalist code-executing agents, OSINT Relay is a strictly sandboxed, read-only intelligence synthesizer. It retains all of our existing robust XML-tag sandboxing to neutralize Indirect Prompt Injections found in hostile social media data.

## 2. Core Architecture Objectives
*   **ChatOps Primary Interface:** Deprioritize the traditional CLI/Web UI. The primary interface will be an async Chat Bot (e.g., Telegram or Discord) where users can issue natural language commands (e.g., *"Monitor @username on Mastodon and alert me of new cryptocurrency domains"*).
*   **Continuous Monitoring (The Watcher):** Transition from run-and-done executions to continuous background monitoring. The agent must wake up periodically, check target profiles via the existing Fetcher architecture, and push asynchronous alerts to the chat if specific conditions are met.
*   **Targeted Multi-Agent LLM Routing:** 
    *   *Agent 1 (Triage/Fast):* A fast, cheap model (e.g., Claude 3.5 Haiku or Llama 3) scans newly fetched posts to see if they match the user's continuous monitoring query.
    *   *Agent 2 (Synthesis/Heavy):* A high-reasoning model (e.g., GPT-4.5, Claude 3 Opus) is only invoked for full psychological/behavioral workups.
*   **Zero-Trust Data Handling:** We will completely reuse the existing `llm.py` injection detection and XML-escaping logic. The agent acts as a secure relay between hostile data and the user.

## 3. Existing Codebase Reusability & Migration Plan
The existing codebase is highly modular and well-organized. We will reuse ~80% of the core engine.

### 🟢 1. Keep As-Is (Core Engine)
*   **`socialosintagent/platforms/*` & `base_fetcher.py`**: The entire platform fetching ecosystem is perfectly structured. Keep rate-limiting logic, standard normalization (`NormalizedPost`), and caching integration.
*   **`socialosintagent/image_processor.py`**: Keep as-is. Resilient image handling is already production-ready.
*   **`socialosintagent/cache.py`**: Keep as-is. File-based caching is excellent for local containerized instances.
*   **`socialosintagent/network_extractor.py`**: Keep as-is. Deterministic network mapping is highly valuable for the background worker to map target associations.
*   **`socialosintagent/prompts/*`**: Keep the existing XML-based security prompts. 

### 🟡 2. Refactor (AI & Orchestration)
*   **`socialosintagent/llm.py`**: 
    *   *Action:* Refactor to support the **Router Pattern**. Add a `run_triage_evaluation()` method to quickly evaluate if a new post triggers a monitoring alert, saving the `run_analysis()` method for deep dives.
*   **`socialosintagent/analyzer.py`**: 
    *   *Action:* Decouple `Console` (Rich printing) dependencies. The analyzer should return pure structured data/events so the ChatOps handler can format them for messaging apps.
*   **`socialosintagent/session_manager.py`**:
    *   *Action:* Expand sessions to include "Monitoring Rules" (e.g., `{ "session_id": "123", "target": "github/torvalds", "condition": "mentions rust", "alert_channel": "telegram_chat_id" }`).

### 🔴 3. Replace/Deprecate (Interfaces)
*   **`socialosintagent/cli_handler.py` & `web_server.py`**: 
    *   *Action:* Move these to a `legacy/` folder or deprecate entirely.
    *   *Replacement:* Create `socialosintagent/chatops_handler.py` utilizing `aiogram` (for Telegram) or `discord.py`.

### 🟣 4. Net-New Components
*   **`socialosintagent/watcher.py` (The Scheduler)**:
    *   *Action:* Implement an `asyncio` background loop. Every *X* minutes, iterate through active "Monitoring Sessions", invoke the fetchers to look for posts newer than the last cached timestamp, run the Triage LLM, and push alerts to the ChatOps handler.

## 4. Required Tech Stack Additions
*   **Chat Integration:** `aiogram` (Telegram) or `discord.py` (Discord) for the primary interface.
*   **Scheduling:** `APScheduler` or native `asyncio` loops for the background watcher.
*   **Database (Optional Upgrade):** While `cache.py` (JSON files) works well, migrating active "Monitoring Rules" to `sqlite3` may be beneficial for concurrency if the background worker and ChatOps bot run in separate threads.

## 5. Security & Guardrails (Strict Enforcement)
*   **Human-In-The-Loop (HITL):** The agent operates in strict read-only mode. It can fetch data and send messages *to the user's private chat*, but it cannot post to social media or interact with targets.
*   **Injection Quarantine:** If `detect_injection_attempt()` in `llm.py` triggers, the agent must quarantine the post, refuse to run the heavy synthesis model on it, and send a specific warning alert to the ChatOps inbox (e.g., *"⚠️ Prompt Injection attempt detected in target's latest Mastodon post."*)

## 6. Implementation Phases (Google Antigravity Instructions)

*   **Phase 1: Interface Swap.** Scaffold `chatops_handler.py`. Connect it to a Telegram Bot token. Prove that a user can text the bot `/analyze twitter/username` and receive the Markdown report as a chat reply. Remove Rich console dependencies from the execution path.
*   **Phase 2: Multi-Agent Routing.** Update `llm.py` to support `TriageAnalyzer` (cheap model) and `SynthesisAnalyzer` (expensive model). 
*   **Phase 3: Continuous Monitoring.** Implement `watcher.py`. Allow the user to text the bot `/monitor bluesky/username for keywords "crypto, wallet"`. Ensure the background loop runs quietly and only pushes messages to the chat when a match occurs.
*   **Phase 4: Dockerization.** Update `docker-compose.yml` to run the bot as a continuous daemon process (`command: python -m socialosintagent.bot`) rather than a run-and-done task.