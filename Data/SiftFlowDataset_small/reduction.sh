#!/bin/bash

classes=( "coast" "forest" "tall" "mountain" "opencountry" )

for i in "${classes[@]}"
do
   rm -R Images/spatial_envelope_256x256_static_8outdoorcategories/${i}*
   rm -R SemanticLabels/labels/${i}*
done
