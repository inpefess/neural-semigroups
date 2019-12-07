#!/bin/bash

SMALLSEMI_VERSION=0.6.11
rm -rf smallsemi*
wget https://www.gap-system.org/pub/gap/gap4/tar.gz/packages/smallsemi-${SMALLSEMI_VERSION}.tar.gz
tar -xvf smallsemi-${SMALLSEMI_VERSION}.tar.gz
rm smallsemi-${SMALLSEMI_VERSION}.tar*
cd smallsemi-${SMALLSEMI_VERSION}
mv data/data2to7 ../smallsemi
cd ../smallsemi
gunzip *
cd ..
rm -rf smallsemi-${SMALLSEMI_VERSION}
