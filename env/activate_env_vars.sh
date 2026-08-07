# prepend conda env lib for GLIBCXX/av compatibility; see docs/DEPLOYMENT_LOG.md #6
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
