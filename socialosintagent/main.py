import argparse
import logging
from dotenv import load_dotenv
import sys
from pathlib import Path

from rich.console import Console

from socialosintagent.analyzer import SocialOSINTAgent
from socialosintagent.cache import CacheManager
from socialosintagent.cli_handler import CliHandler
from socialosintagent.client_manager import ClientManager
from socialosintagent.llm import LLMAnalyzer

logger = logging.getLogger("SocialOSINTAgent.main")

def main():
    load_dotenv()

    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    log_file_path = logs_dir / "analyzer.log"

    parser = argparse.ArgumentParser(
        description="Social Media OSINT analyzer using LLMs...",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Environment Variables Required... (rest of epilog is unchanged)
""",
    )
    parser.add_argument("--stdin", action="store_true", help="Read analysis request from stdin as JSON.")
    parser.add_argument("--format", choices=["json", "markdown"], default="markdown", help="Output format for saving analysis reports.")
    parser.add_argument("--no-auto-save", action="store_true", help="Disable automatic saving of reports.")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], default="WARNING", help="Set the logging level.")
    parser.add_argument("--offline", action="store_true", help="Run in offline mode, using only cached data.")
    parser.add_argument("--unsafe-allow-external-media", action="store_true", help="Security: Allow downloading media from domains outside known social media CDNs (e.g. personal servers).")
    args = parser.parse_args()

    log_level_numeric = getattr(logging, args.log_level.upper())
    
    logging.basicConfig(level=log_level_numeric, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", handlers=[logging.FileHandler(log_file_path), logging.StreamHandler()])
    
    logging.getLogger("SocialOSINTAgent").setLevel(log_level_numeric)
    
    if args.offline: logging.info("Running in OFFLINE mode.")

    try:
        base_dir = Path("data")
        cache_manager = CacheManager(base_dir, args.offline)
        llm_analyzer = LLMAnalyzer(args.offline)
        client_manager = ClientManager(args.offline)

        # The agent is the core logic engine
        agent_instance = SocialOSINTAgent(args, cache_manager, llm_analyzer, client_manager)
        
        if args.stdin:
            # Non-interactive mode
            agent_instance.process_stdin()
        else:
            # Interactive mode
            cli = CliHandler(agent_instance, args)
            cli.run()

    except RuntimeError as e:
        error_console = Console(stderr=True, style="bold red")
        error_console.print(f"\nCRITICAL ERROR: {e}")
        error_console.print("Ensure necessary API keys and platform credentials/URLs are correctly set.")
        sys.exit(1)
    except Exception as e:
        logging.getLogger("SocialOSINTAgent").critical(f"An unexpected critical error occurred: {e}", exc_info=True)
        error_console = Console(stderr=True, style="bold red")
        error_console.print(f"\nUNEXPECTED CRITICAL ERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()