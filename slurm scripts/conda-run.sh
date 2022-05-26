#!/bin/bash
# Activate a conda environment on a node and run a command
# This script takes a sequence of arguments, which will be executed consecutively

export LC_ALL=C.UTF-8
export LANG=C.UTF-8

# dirty hack: https://github.com/conda/conda/issues/9392#issuecomment-617345019 
unset CONDA_SHLVL
# initialize conda
source "$(conda info --base)""/etc/profile.d/conda.sh"

# Use this to suppress error codes that may interrupt the job or interfere with reporting.
# Do show the user that an error code has been received and is being suppressed.
# see https://stackoverflow.com/questions/11231937/bash-ignoring-error-for-a-particular-command 
eval "${@}" || echo "Exit with error code $? (suppressed)"