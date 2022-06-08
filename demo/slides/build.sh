#!/bin/bash
set -e

for slide in $(cat order); do
  inkscape --file=$slide.svg --without-gui --export-pdf=$slide.pdf &
done
wait

(cat order; echo 'demo') | sed -e 's/$/.pdf/g' | xargs pdfunite

for slide in $(cat order); do
  rm $slide.pdf
done
