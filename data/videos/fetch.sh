#!/bin/bash
# Pexels serves these behind Cloudflare, so they cannot be curl'ed blind.
# Open each page, press "Free Download", and save the file into this folder
# under the name in manifest.csv.
cd "$(dirname "$0")" || exit 1
printf '%-10s %-6s %s\n' CLIP ROLE PAGE
tail -n +2 manifest.csv | while IFS=, read -r id role scene page name; do
    if [ -f "$name" ]; then
        printf '%-10s %-6s already here (%s)\n' "$id" "$role" "$name"
    else
        printf '%-10s %-6s %s\n' "$id" "$role" "$page"
    fi
done
