#!/bin/bash

FILE=$1

perl -pi -e 's/U/u/g' $FILE
perl -pi -e 's/sin\(A\)/sin_alpha/g' $FILE
perl -pi -e 's/cos\(A\)/cos_alpha/g' $FILE
perl -pi -e 's/sin_alpha\*\*2/sin_alpha_2/g' $FILE
perl -pi -e 's/u([0-2])\*\*2/u$1_2/g' $FILE
perl -pi -e 's/u\*\*([0-9])/u_$1/g' $FILE
perl -pi -e 's/dpdu0/dp0du0/g' $FILE
perl -pi -e 's/dpdu1/dp1du0/g' $FILE
perl -pi -e 's/dpdu2/dp2du0/g' $FILE
perl -pi -e 's/dpdu3/dp0du1/g' $FILE
perl -pi -e 's/dpdu4/dp1du1/g' $FILE
perl -pi -e 's/dpdu5/dp2du1/g' $FILE
perl -pi -e 's/dpdu6/dp0du2/g' $FILE
perl -pi -e 's/dpdu7/dp1du2/g' $FILE
perl -pi -e 's/dpdu8/dp2du2/g' $FILE
