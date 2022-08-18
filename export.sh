#!/usr/bin/env bash

. ./build.sh

docker save pssn_noduledetector | gzip -c > pssn.tar.gz
