# !/bin/sh

apptainer exec --nv --bind .:/workspace apptainer/videoqa-inference.sif uv run main.py
