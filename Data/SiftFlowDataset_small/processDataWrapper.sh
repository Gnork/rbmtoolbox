#!/bin/bash

LABELS_SOURCES_DIR="SemanticLabels/spatial_envelope_256x256_static_8outdoorcategories"
LABELS_DIR="SemanticLabels/labels"
NAMES_DIR="SemanticLabels/classes"

[ -d ${NAMES_DIR} ] && rm -R ${NAMES_DIR}
mkdir ${NAMES_DIR}

[ -d ${LABELS_DIR} ] && rm -R ${LABELS_DIR}
mkdir ${LABELS_DIR}

for file in ${LABELS_SOURCES_DIR}/*.mat; do
	filename=$(basename ${file})
	labelsOut="${LABELS_DIR}/${filename}"
	namesOut="${NAMES_DIR}/${filename}"
	octave --silent --eval "processData(\"$file\", \"$labelsOut\", \"$namesOut\")"
done
