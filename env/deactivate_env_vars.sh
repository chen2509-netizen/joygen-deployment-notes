# reverse activate_env_vars.sh; without this, LD_LIBRARY_PATH leaks into
# the shell after `conda deactivate` and breaks system tools (e.g. git)
unset LD_LIBRARY_PATH
unset CUDA_HOME
