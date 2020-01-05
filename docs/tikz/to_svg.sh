#!/usr/bin/env bash

# This works with Inkscape 1.0 beta.
# We use Poppler because otherwise math text looks like Korean.
flatpak run org.inkscape.Inkscape \
    --without-gui --pdf-poppler --export-area-drawing pencils.pdf

# Remove "height" and "width" items so that the image takes all the place it is
# given.
sed -i -E -e 's/^\s+(height|width)=".*"//' pencils.svg

# When Inkscape uses Poppler to convert PDF to SVG, colours are slightly
# modified, probably because Poppler works with RGB instead of HTML definitions,
# and in the end there is some approximation (truncation?) error.
#
# We make sure that the true Julia colours
# (https://github.com/JuliaLang/julia-logo-graphics#color-definitions) are in
# the final picture.
julia_green=389826
inkscape_green=359723
julia_purple=9558b2
inkscape_purple=9457b0

# First make sure that "Inkscape" colours are present.
for c in $inkscape_green $inkscape_purple; do
    if ! grep -q $c pencils.svg; then
        echo "Error: colour #$c not found!"
        exit 1
    fi
done

# Replace colours.
sed -i -e "s/#$inkscape_green/#$julia_green/
           s/#$inkscape_purple/#$julia_purple/" pencils.svg

mv -v pencils.svg ../src/img/
