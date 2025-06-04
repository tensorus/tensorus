#!/usr/bin/env bash
# Setup script for Tensorus development environment
set -e

# Install Python dependencies
pip install -r requirements.txt
pip install -r requirements-test.txt

# Install Node.js dependencies for the MCP server (used in integration tests)
pushd mcp_tensorus_server
npm install
popd
