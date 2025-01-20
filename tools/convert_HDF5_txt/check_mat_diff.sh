#!/bin/bash

MAT_A=$1
MAT_B=$2

# check difference between two matrices


paste $MAT_A $MAT_B | awk 'BEGIN{ai=0;mi=0;aj=0;mj=0;d=0;p=0}
{
if($1-$8!=0.) mi++;
if($2-$9!=0.) ai++;
if($3-$10!=0.) mj++;
if($4-$11!=0.) aj++;
if(sqrt(($5-$12)**2)>0.00001) d++;
if(sqrt(($6-$13)**2)>0.00001) p++;
if(sqrt(($7-$14)**2)>0.00001) c ++;
}
END{
if(ai> 0||aj>0||mi>0||mj>0||d>0||p>0||c>0 ) print "Different elements: ", ai, aj, mi, mj, d, p, c;
else print "Matrices are identical";
}'