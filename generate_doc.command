#! /bin/bash

cd $PROJ/classic_rl

pdoc --html --output-dir docs modules --force --config latex_math=True
cd docs
cp -a modules/. .
rm -rf modules
cd ..