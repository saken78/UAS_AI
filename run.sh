#!/usr/bin/env bash
cd "$(dirname "$0")"
export LD_LIBRARY_PATH="/nix/store/n35z8vvlr7c5k1406n5bwd0f8h2hgj1j-gcc-15.2.0-lib/lib:$LD_LIBRARY_PATH"
exec .venv/bin/streamlit run app.py --server.port 8501
