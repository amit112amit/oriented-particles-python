#!/bin/bash

FILE=$1

perl -pi -e 's/\\//g' $FILE
perl -i -e 'undef $/; $_=<>; s/\n\s+\*\*/**/g; print' $FILE
perl -pi -e 's/v([ij])_mag\*\*2/v\1_mag_sqr/g' $FILE
perl -pi -e 's/v([ij])_mag\*\*([0-9])/pow(v\1_mag,\2)/g' $FILE
perl -pi -e 's/sin_alpha_([ij])\*\*2/sin_alpha_\1_2/g' $FILE
perl -pi -e 's/cos_alpha_([ij])\*\*2/cos_alpha_\1_2/g' $FILE
perl -pi -e 's/sin_alpha_([ij])\*\*([0-9])/pow(sin_alpha_\1,\2)/g' $FILE
perl -pi -e 's/cos_alpha_([ij])\*\*([0-9])/pow(cos_alpha_\1,\2)/g' $FILE
perl -pi -e 's/([a-z0-9]+)\*\*2/\1*\1/g' $FILE
perl -pi -e 's/([a-z0-9]+)\*\*([0-9])/pow(\1,\2)/g' $FILE
