#!/usr/bin/env bash
# Builds app-mac/ in release mode and packages it as /Applications/Semse.app.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
APP_MAC="$ROOT/app-mac"
DIST="$APP_MAC/dist"
BUNDLE="$DIST/Semse.app"
PYTHON="$ROOT/.venv/bin/python"

echo "── Building (release) ──"
(cd "$APP_MAC" && swift build -c release)

echo "── Assembling Semse.app ──"
rm -rf "$BUNDLE"
mkdir -p "$BUNDLE/Contents/MacOS" "$BUNDLE/Contents/Resources"
cp "$APP_MAC/.build/release/Memory" "$BUNDLE/Contents/MacOS/Semse"

cat > "$BUNDLE/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleIdentifier</key>
    <string>com.projectithaca.semse</string>
    <key>CFBundleName</key>
    <string>Semse</string>
    <key>CFBundleDisplayName</key>
    <string>Semse</string>
    <key>CFBundleExecutable</key>
    <string>Semse</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0</string>
    <key>CFBundleIconFile</key>
    <string>AppIcon</string>
    <key>LSMinimumSystemVersion</key>
    <string>26.0</string>
    <key>LSUIElement</key>
    <true/>
    <key>NSHighResolutionCapable</key>
    <true/>
</dict>
</plist>
PLIST

echo "── Generating icon ──"
mkdir -p "$APP_MAC/AppIconSource"
"$PYTHON" - "$APP_MAC/AppIconSource/icon_1024.png" <<'PYEOF'
import sys
from PIL import Image, ImageDraw

out = sys.argv[1]
S = 1024
img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
draw = ImageDraw.Draw(img)

top = (48, 43, 99)      # deep indigo
bottom = (139, 92, 246) # violet
gradient = Image.new("RGBA", (S, S))
gd = ImageDraw.Draw(gradient)
for y in range(S):
    t = y / (S - 1)
    r = int(top[0] + (bottom[0] - top[0]) * t)
    g = int(top[1] + (bottom[1] - top[1]) * t)
    b = int(top[2] + (bottom[2] - top[2]) * t)
    gd.line([(0, y), (S, y)], fill=(r, g, b, 255))

# macOS icon grid: content inset ~10%, continuous-corner radius ~22.5%
inset = int(S * 0.10)
radius = int(S * 0.225)
mask = Image.new("L", (S, S), 0)
ImageDraw.Draw(mask).rounded_rectangle(
    [inset, inset, S - inset, S - inset], radius=radius, fill=255
)
img.paste(gradient, (0, 0), mask)

# Magnifying glass: circle upper-left of center, handle to lower-right
draw = ImageDraw.Draw(img)
stroke = int(S * 0.055)
cx, cy = int(S * 0.46), int(S * 0.46)
cr = int(S * 0.165)
draw.ellipse(
    [cx - cr, cy - cr, cx + cr, cy + cr],
    outline=(255, 255, 255, 255), width=stroke,
)
import math
angle = math.radians(45)
hx0 = cx + int((cr - stroke * 0.25) * math.cos(angle))
hy0 = cy + int((cr - stroke * 0.25) * math.sin(angle))
hx1 = int(S * 0.665)
hy1 = int(S * 0.665)
draw.line([(hx0, hy0), (hx1, hy1)], fill=(255, 255, 255, 255), width=stroke)
hr = stroke // 2
draw.ellipse([hx1 - hr, hy1 - hr, hx1 + hr, hy1 + hr], fill=(255, 255, 255, 255))

img.save(out)
print(f"wrote {out}")
PYEOF

ICONSET="$DIST/AppIcon.iconset"
rm -rf "$ICONSET"
mkdir -p "$ICONSET"
SRC="$APP_MAC/AppIconSource/icon_1024.png"
for size in 16 32 128 256 512; do
  sips -z "$size" "$size" "$SRC" --out "$ICONSET/icon_${size}x${size}.png" >/dev/null
  double=$((size * 2))
  sips -z "$double" "$double" "$SRC" --out "$ICONSET/icon_${size}x${size}@2x.png" >/dev/null
done
iconutil -c icns "$ICONSET" -o "$BUNDLE/Contents/Resources/AppIcon.icns"

echo "── Signing (ad-hoc) ──"
codesign --force --deep -s - "$BUNDLE"

echo "── Installing to /Applications ──"
rm -rf /Applications/Semse.app
cp -R "$BUNDLE" /Applications/Semse.app

echo "Done: /Applications/Semse.app"
