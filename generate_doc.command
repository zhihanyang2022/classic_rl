#! /bin/bash

pdoc --html --output-dir docs .
cd docs
cp -a classic_rl/. .
rm -rf classic_rl
cd ..