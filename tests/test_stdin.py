#!/usr/bin/env python3
"""
Test cases for the improved process_stdin() method.
Demonstrates comprehensive error handling and validation.
"""

import json
import subprocess
import sys
from pathlib import Path

# ANSI color codes for pretty output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"

def run_test(test_name: str, input_json: dict, expected_exit_code: int, description: str):
    """Run a single test case."""
    print(f"\n{BLUE}{'='*70}{RESET}")
    print(f"{BLUE}Test: {test_name}{RESET}")
    print(f"{YELLOW}Description: {description}{RESET}")
    print(f"Input JSON:")
    print(json.dumps(input_json, indent=2))
    print(f"\nExpected exit code: {expected_exit_code}")
    
    # Simulate running the command
    # In real usage: echo '...' | python -m socialosintagent.main --stdin
    json_str = json.dumps(input_json)
    
    print(f"\n{YELLOW}Command would be:{RESET}")
    print(f"echo '{json_str}' | python -m socialosintagent.main --stdin")
    print(f"{BLUE}{'='*70}{RESET}\n")


def main():
    """Run all test cases."""
    
    print(f"{GREEN}{'#'*70}")
    print("# Improved process_stdin() - Test Suite")
    print(f"{'#'*70}{RESET}\n")
    
    # Test 1: Valid request (success case)
    run_test(
        test_name="Valid Request",
        input_json={
            "platforms": {
                "twitter": ["elonmusk", "naval"],
                "reddit": ["spez"]
            },
            "query": "What are the primary interests and communication patterns of these users?",
            "fetch_options": {"default_count": 25}
        },
        expected_exit_code=0,
        description="A properly formatted request with all required fields"
    )
    
    # Test 2: Invalid JSON syntax
    run_test(
        test_name="Invalid JSON Syntax",
        input_json="This is not valid JSON (imagine malformed brackets)",
        expected_exit_code=1,
        description="Malformed JSON should return structured error with line/column info"
    )
    
    # Test 3: Missing 'platforms' field
    run_test(
        test_name="Missing 'platforms' Field",
        input_json={
            "query": "What are their interests?"
        },
        expected_exit_code=1,
        description="Missing required 'platforms' field should provide helpful example"
    )
    
    # Test 4: Missing 'query' field
    run_test(
        test_name="Missing 'query' Field",
        input_json={
            "platforms": {"twitter": ["user1"]}
        },
        expected_exit_code=1,
        description="Missing required 'query' field should provide helpful example"
    )
    
    # Test 5: Wrong type for 'platforms' (should be dict)
    run_test(
        test_name="Invalid Type for 'platforms'",
        input_json={
            "platforms": ["twitter", "reddit"],  # Should be dict, not list
            "query": "What are their interests?"
        },
        expected_exit_code=1,
        description="Type mismatch should specify expected vs received types"
    )
    
    # Test 6: Wrong type for 'query' (should be string)
    run_test(
        test_name="Invalid Type for 'query'",
        input_json={
            "platforms": {"twitter": ["user1"]},
            "query": 12345  # Should be string, not number
        },
        expected_exit_code=1,
        description="Type mismatch for query should provide clear error"
    )
    
    # Test 7: Empty 'query' string
    run_test(
        test_name="Empty Query String",
        input_json={
            "platforms": {"twitter": ["user1"]},
            "query": "   "  # Whitespace-only string
        },
        expected_exit_code=1,
        description="Empty or whitespace-only query should be rejected"
    )
    
    # Test 8: Empty 'platforms' dict
    run_test(
        test_name="Empty Platforms Dict",
        input_json={
            "platforms": {},  # No platforms specified
            "query": "What are their interests?"
        },
        expected_exit_code=1,
        description="Empty platforms dict should be rejected"
    )
    
    # Test 9: Unconfigured platforms
    run_test(
        test_name="Unconfigured Platforms",
        input_json={
            "platforms": {
                "instagram": ["user1"],  # Not supported
                "linkedin": ["user2"]    # Not supported
            },
            "query": "What are their interests?"
        },
        expected_exit_code=1,
        description="All platforms unconfigured should fail with helpful message"
    )
    
    # Test 10: Mix of valid and invalid platforms
    run_test(
        test_name="Mixed Valid/Invalid Platforms",
        input_json={
            "platforms": {
                "twitter": ["elonmusk"],
                "instagram": ["user1"],  # Not supported
                "reddit": ["spez"]
            },
            "query": "What are their interests?"
        },
        expected_exit_code=0,
        description="Should skip invalid platforms and proceed with valid ones"
    )
    
    # Test 11: Invalid usernames type (should be list)
    run_test(
        test_name="Invalid Usernames Type",
        input_json={
            "platforms": {
                "twitter": "elonmusk"  # Should be list, not string
            },
            "query": "What are their interests?"
        },
        expected_exit_code=1,
        description="Usernames should be a list, not a string"
    )
    
    # Test 12: Empty usernames list
    run_test(
        test_name="Empty Usernames List",
        input_json={
            "platforms": {
                "twitter": []  # No usernames
            },
            "query": "What are their interests?"
        },
        expected_exit_code=1,
        description="Empty usernames list should be rejected"
    )
    
    # Test 13: Usernames with only whitespace
    run_test(
        test_name="Whitespace-Only Usernames",
        input_json={
            "platforms": {
                "twitter": ["  ", "\t", "\n"]  # All whitespace
            },
            "query": "What are their interests?"
        },
        expected_exit_code=1,
        description="After sanitization, should reject if no valid usernames remain"
    )
    
    # Test 14: Valid with no-auto-save flag
    run_test(
        test_name="Valid Request with --no-auto-save",
        input_json={
            "platforms": {"hackernews": ["pg"]},
            "query": "What are their technical interests?"
        },
        expected_exit_code=0,
        description="With --no-auto-save flag, should print report to stdout"
    )
    
    # Test 15: Valid with JSON output format
    run_test(
        test_name="Valid Request with JSON Format",
        input_json={
            "platforms": {"github": ["torvalds"]},
            "query": "What are their recent activities?"
        },
        expected_exit_code=0,
        description="With --format json flag, should output structured JSON"
    )
    
    print(f"\n{GREEN}{'#'*70}")
    print("# Test Suite Complete")
    print(f"{'#'*70}{RESET}\n")
    
    print(f"{YELLOW}Expected Error Outputs:{RESET}\n")
    
    print("1. Invalid JSON:")
    print(json.dumps({
        "error": "Invalid JSON",
        "message": "Expecting property name...",
        "line": 1,
        "column": 15,
        "help": "Ensure your JSON is properly formatted..."
    }, indent=2))
    
    print("\n2. Missing Required Field:")
    print(json.dumps({
        "error": "Missing required fields",
        "missing_fields": ["query"],
        "provided_fields": ["platforms"],
        "example": {
            "platforms": {"twitter": ["example_user"]},
            "query": "What are their interests?"
        }
    }, indent=2))
    
    print("\n3. Type Mismatch:")
    print(json.dumps({
        "error": "Invalid field type",
        "field": "platforms",
        "expected_type": "dict",
        "received_type": "list",
        "example": {"twitter": ["user1", "user2"]}
    }, indent=2))
    
    print(f"\n{GREEN}Success Output (with auto-save):{RESET}")
    print(json.dumps({
        "success": True,
        "output_file": "data/outputs/analysis_20260202_120000_twitter_What_are_their.md",
        "metadata": {
            "query": "What are their interests?",
            "targets": {"twitter": ["user1"]},
            "generated_utc": "2026-02-02 12:00:00 UTC"
        }
    }, indent=2))
    
    print(f"\n{GREEN}Success Output (without auto-save, markdown):{RESET}")
    print("# OSINT Analysis Report\n\n**Query:** `What are their interests?`\n...")
    
    print(f"\n{GREEN}Success Output (without auto-save, JSON):{RESET}")
    print(json.dumps({
        "success": True,
        "metadata": {"query": "..."},
        "report": "# OSINT Analysis Report\n\n..."
    }, indent=2))


if __name__ == "__main__":
    main()