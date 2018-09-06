#!/bin/bash -x 
# 
# This is just a wrapper for http://entrproject.org/
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
GPG_TTY=$(tty)
export GPG_TTY
cd "$DIR" || exit 1
# source venv/bin/activate

# Can also use `find` here if silversurfer not installed
# /usr/bin/ag -l | /usr/bin/entr -d bash "$DIR/buildDocs.sh"

pip3 install --user sphinx-autobuild
sphinx-autobuild "$DIR" "$DIR/_build/html"
