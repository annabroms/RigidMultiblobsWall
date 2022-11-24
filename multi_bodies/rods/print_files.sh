#!/bin/bash

target="/home/a/n/annabrom/NOBACKUP/RigidMultiblobsWall/multi_bodies/rods/dynamic_rods_T1_N10
$1"

pushd "$target" > /dev/null
let count=0
for f in *
do
    echo $f
    let count=count+1
done
popd
echo ""
echo "Count: $count"
