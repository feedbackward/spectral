#!/bin/bash

## A simple loop over all the datasets specified.
for arg
do
    bash remote_learn_off_run.sh "${arg}"
    bash remote_learn_default_run.sh "${arg}"
    bash remote_learn_fast_run.sh "${arg}"
done
