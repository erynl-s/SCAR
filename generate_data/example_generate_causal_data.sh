#!/bin/bash

# get sampling distributions for ISV and cause and effect variables

python define_causal_graph.py

# ----------------

#generate data with previously defined effect distributions
exp= #experiment name you used in define_causal_graph.py

echo generating data for $exp

# t_rois defines the targeted or effected region(s)

python generate_causal_data.py \
  --t_rois "left lateral ventricle" \
  --c_graph "collider" \
  --isv 1 \
  --effect 1 \
  --expname "$exp"
