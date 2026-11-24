"""Lightweight HTTP client and MCP CLI for `tensorus`."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from tensorus import __version__


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tensorus",
        description="Tensorus Python SDK & MCP client CLI.",
    )
    parser.add_argument(
        "--version", action="store_true", help="Print installed tensorus SDK version."
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("mcp", help="Run the Tensorus Model Context Protocol (MCP) server over stdio.")

    p_whoami = sub.add_parser("whoami", help="Inspect the authenticated workspace identity.")
    p_whoami.add_argument(
        "--url", default=os.environ.get("TENSORUS_URL"), help="Tensorus base URL."
    )
    p_whoami.add_argument("--api-key", default=os.environ.get("TENSORUS_API_KEY"), help="API key.")

    p_ds = sub.add_parser("datasets", help="List accessible datasets in the workspace.")
    p_ds.add_argument("--url", default=os.environ.get("TENSORUS_URL"), help="Tensorus base URL.")
    p_ds.add_argument("--api-key", default=os.environ.get("TENSORUS_API_KEY"), help="API key.")

    p_npy = sub.add_parser("inspect-npy", help="Validate and inspect a local .npy header.")
    p_npy.add_argument("path", type=Path, help="Path to a local .npy file.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.version:
        print(__version__)
        return 0
    if args.command == "mcp":
        from tensorus.mcp_server import run

        run()
        return 0
    if args.command in ("whoami", "datasets"):
        if not args.url or not args.api_key:
            msg = "Error: provide --url / --api-key or set TENSORUS_URL and TENSORUS_API_KEY."
            print(msg, file=sys.stderr)
            return 2
        from tensorus.client import Client

        with Client(args.url, args.api_key) as client:
            payload = client.me() if args.command == "whoami" else client.datasets()
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if args.command == "inspect-npy":
        from tensorus.serialization import read_npy_header

        size = args.path.stat().st_size
        with args.path.open("rb") as handle:
            shape, fortran_order, dtype = read_npy_header(handle, size, size)
            offset = handle.tell()
        info = {
            "path": str(args.path),
            "dtype": str(dtype),
            "shape": list(shape),
            "fortran_order": fortran_order,
            "data_offset": offset,
        }
        print(json.dumps(info, indent=2))
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
