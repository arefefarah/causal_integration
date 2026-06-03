#!/usr/bin/env python
"""Thin wrapper around ``causal_msi.cli.train`` (see that module)."""

from __future__ import annotations

from causal_msi.cli.train import app

if __name__ == "__main__":
    app()
