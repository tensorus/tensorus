#!/usr/bin/env python3
"""
Tensorus API Key Generator

A simple command-line utility to generate secure API keys for Tensorus.
Use this tool to create API keys for development, testing, and production.

Usage:
    python generate_api_key.py                    # Generate 1 key
    python generate_api_key.py --count 5          # Generate 5 keys
    python generate_api_key.py --help             # Show help
"""

import argparse
import sys
from tensorus.auth.key_generator import generate_api_key, TensorusAPIKey


def main():
    parser = argparse.ArgumentParser(
        description="Generate secure API keys for Tensorus authentication",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_api_key.py                    # Generate 1 key
  python generate_api_key.py --count 3          # Generate 3 keys
  python generate_api_key.py --format env       # Generate in .env format

Generated keys can be used in:
  - Environment variables: TENSORUS_API_KEYS=key1,key2,key3
  - Bearer token headers: Authorization: Bearer <key>
  - MCP server configuration: --mcp-api-key <key>
        """
    )
    
    parser.add_argument(
        "--count", "-c",
        type=int,
        default=1,
        help="Number of API keys to generate (default: 1)"
    )
    
    parser.add_argument(
        "--format", "-f",
        choices=["plain", "env", "json"],
        default="plain",
        help="Output format: plain, env, or json (default: plain)"
    )
    
    parser.add_argument(
        "--prefix", "-p",
        default="",
        help="Optional prefix for key names in env/json format"
    )
    
    args = parser.parse_args()
    
    if args.count < 1:
        print("Error: Count must be at least 1", file=sys.stderr)
        return 1
    
    if args.count > 100:
        print("Error: Cannot generate more than 100 keys at once", file=sys.stderr)
        return 1
    
    # Generate keys
    keys = [generate_api_key() for _ in range(args.count)]
    
    # Output in requested format
    if args.format == "plain":
        if args.count == 1:
            print(f"Generated API key: {keys[0]}")
            print(f"Masked format:     {TensorusAPIKey.mask_key(keys[0])}")
            print(f"Format valid:      {TensorusAPIKey.validate_format(keys[0])}")
        else:
            print(f"Generated {args.count} API keys:")
            for i, key in enumerate(keys, 1):
                print(f"{i:2d}. {key}")
    
    elif args.format == "env":
        if args.count == 1:
            print(f"TENSORUS_API_KEYS={keys[0]}")
        else:
            key_list = ",".join(keys)
            print(f"TENSORUS_API_KEYS={key_list}")
        
        print("\n# Add this to your .env file or set as environment variable")
        print("# For production, store securely and don't commit to version control")
    
    elif args.format == "json":
        import json
        if args.count == 1:
            data = {
                "api_key": keys[0],
                "masked": TensorusAPIKey.mask_key(keys[0]),
                "format_valid": TensorusAPIKey.validate_format(keys[0])
            }
        else:
            data = {
                "api_keys": keys,
                "count": len(keys),
                "masked": [TensorusAPIKey.mask_key(key) for key in keys]
            }
        
        print(json.dumps(data, indent=2))
    
    # Additional usage examples for first run
    if args.format == "plain" and args.count == 1:
        print("\nUsage examples:")
        print(f"  Environment:  export TENSORUS_API_KEYS={keys[0]}")
        print(f"  cURL:         curl -H 'Authorization: Bearer {keys[0]}' ...")
        print(f"  Python:       headers = {{'Authorization': 'Bearer {keys[0]}'}}")
        print(f"  MCP Server:   python -m tensorus.mcp_server --mcp-api-key {keys[0]}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())