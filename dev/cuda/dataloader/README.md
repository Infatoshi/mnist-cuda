# Dataloader
- if we compare the `dataloaderv2.py` script in python to our dataloader in CUDA, we already get ~40% speedup in loading the data.
- feel free to run yourself
    - `python dataloaderv2.py`
    - `nvcc -o v1 main.cu dataloader/dataloader.cu` then `./v1`
    - CUDA takes ~5.2e-5 seconds per batch, python takes ~8.3e-5 seconds per batch

## TODO
