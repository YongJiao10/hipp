#!/bin/bash
source /opt/conda/bin/activate hippo2
# We downgrade snakemake instead of updating base conda
pip install "snakemake<8.0.0" "snakemake-minimal<8.0.0"
