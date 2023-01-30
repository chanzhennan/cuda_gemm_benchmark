#!/bin/bash

# PWD must be /checkout
echo "workdir: $PWD"

# add safe.directory for git
git config --global --add safe.directory /checkout

# install pre-commit
pre-commit install

echo -e "show PR_CHANGED_FILES"
echo $PR_CHANGED_FILES

# run pre-commit
echo -e "start run pre-commit \n\n"
echo $PR_CHANGED_FILES | xargs pre-commit run --show-diff-on-failure --files
