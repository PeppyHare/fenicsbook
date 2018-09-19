#!/bin/bash -x 
# 
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
GPG_TTY=$(tty)
export GPG_TTY
cd "$DIR" || exit 1

# un-comment in case of virtualenv
# source venv/bin/activate

# Can also use `find` here if silversurfer not installed
# This is just a wrapper for http://entrproject.org/
# /usr/bin/ag -l | /usr/bin/entr -d bash "$DIR/buildDocs.sh"

pip3 install --user sphinx-autobuild
sphinx-autobuild -p 30001 -H "0.0.0.0" -i ".git/*" -i "docs/*" "$DIR" "$DIR/docs"
