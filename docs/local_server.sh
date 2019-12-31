#!/usr/bin/env bash

cd build || exit 1
python3 -m http.server --bind localhost
