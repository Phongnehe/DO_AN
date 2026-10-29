#!/bin/sh
cd "$(dirname "$0")" || exit 1
exec /Applications/Xcode.app/Contents/Developer/usr/bin/python3 scripts/local_runtime.py --streamlit --server.address 127.0.0.1
