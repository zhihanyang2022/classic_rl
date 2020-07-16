#! /bin/bash

cd $PROJ/classic_rl

pdoc --html --output-dir docs . --force
cd docs
cp -a classic_rl/. .
rm -rf classic_rl
cd ..