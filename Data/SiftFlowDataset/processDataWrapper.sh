#!/bin/bash

LABELS_DIR="SemanticLabels/labels"
NAMES_DIR="SemanticLabels/names"

[ -d ${NAMES_DIR} ] && rm -R ${NAMES_DIR}
mkdir ${NAMES_DIR}

[ -d ${LABELS_DIR} ] && rm -R ${LABELS_DIR}
mkdir ${LABELS_DIR}

for file in SemanticLabels/spatial_envelope_256x256_static_8outdoorcategories/*.mat; do
	filename=$(basename ${file})
	labelsOut="${LABELS_DIR}/${filename}"
	namesOut="${NAMES_DIR}/${filename}"
	octave --silent --eval "processData(\"$file\", \"$labelsOut\", \"$namesOut\")"
done
