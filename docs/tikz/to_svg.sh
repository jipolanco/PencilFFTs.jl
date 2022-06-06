#!/usr/bin/env bash

# This works with Inkscape 1.0 beta.
# We use Poppler because otherwise math text looks like Korean.
inkscape --pdf-poppler --export-area-drawing pencils.pdf -o pencils.svg

# Remove "height" and "width" items so that the image takes all the place it is
# given.
sed -i -E -e 's/^\s+(height|width)=".*"//' pencils.svg

mv -v pencils.svg ../src/img/
