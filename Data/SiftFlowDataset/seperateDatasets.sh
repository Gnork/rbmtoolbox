#!/bin/bash
([a-zA-Z]*)[0-9]*.mat
for file in ${LABELS_SOURCES_DIR}/*.mat; do
	filename=$(basename ${file})
