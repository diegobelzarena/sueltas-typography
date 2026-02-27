#!/usr/bin/env bash
# Download pre‑trained CharNet weights.  The remote URL has been known to
# disappear; see the README for an alternate mirror if this fails.
mkdir -p weights

wget https://cloudstor.aarnet.edu.au/plus/s/c0PaY4pzPUhPmL9/download \
    -O weights/icdar2015_hourglass88.pth
