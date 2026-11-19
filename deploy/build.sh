#!/bin/sh
# Builds the static site into a FRESH directory (default dist/) for Cloudflare Pages production deploys.
# Local JS/CSS go under a content-hashed directory so a new index.html can never load modules cached
# from an older deploy (the zone gives assets a 4 h browser TTL). Hashed files are immutable.
set -eu
cd "$(dirname "$0")/.."
out=${1:-dist}
if [ -e "$out" ] || [ -L "$out" ]; then printf 'build: output already exists; choose a fresh directory\n' >&2; exit 1; fi
assets="style.css s1.css app.js cache.js decide.js engine.js logits.js s1.js dev/stub-engine.js"
v=$(cat $assets | shasum -a 256 | cut -c1-12)
mkdir -p "$out/$v/dev"
for f in $assets; do cp "$f" "$out/$v/$f"; done
for page in index.html decide.html; do
  cp "$page" "$out/$page"
  for f in $assets; do
    sed -E -e "s#(src|href)=\"(\./)?$f\"#\1=\"./$v/$f\"#g" "$out/$page" > "$out/$page.tmp"
    mv "$out/$page.tmp" "$out/$page"
  done
done
if grep -oE '(src|href)="[^"]*\.(js|css)"' "$out/index.html" "$out/decide.html" | grep -qv "/$v/"; then
  printf 'build: asset path rewrite failed\n' >&2; exit 1
fi
printf '/%s/*\n  Cache-Control: public, max-age=31536000, immutable\n' "$v" > "$out/_headers"
