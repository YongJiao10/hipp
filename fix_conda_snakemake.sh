#!/bin/bash
source /opt/conda/bin/activate hippo2
# Update conda inside the environment or downgrade snakemake to work with older conda
conda install -y -c conda-forge "conda>=24.7.1" || conda install -y -c conda-forge snakemake=7.32.4
