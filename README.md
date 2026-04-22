# What is this Repo?

Using the GUI:
- Click "Add" then click two points on the 3DGS model
- Internally:
    - Draw a ray from A - B
    - Apply ball-point query to reduce computation
    - Progressively Ray March
    - At each step draw N samples uniformly and perpendicular to AB
    - Progressively Ray March to find the nearest "surface"
    - Select the closest surface point
    - save and display

## What is a surface? (Our interpretations)
A sample on some ray, where the two neighbouring samples have density > threshold and density < threshold


# Installation

This has been tested on NVidia RTX 3090 w/ py3.10 pt2.4 and cu11.8, and NVidia RTX 4090 w/ py3.10 pt2.4 and cu12.4

1. Create a conda environment with `conda create -n vsres python=3.10` (pytorch version requires compatibility with `gsplat`; I use py310)
2. Install pytorch (requires compatibility with `gsplat`; I use pt24)
3. Download `gsplat` either via `pip install gsplat` or with wheel (I used `gsplat-1.5.3+pt24cu124-cp310-cp310-linux_x86_64.whl`)
4. Run `pip install -r requirements.txt`
5. (Optional) Cry because that probably didn't work and the guy who created this repo also has no clue, so you choose not to start an issue because there's no way he's resolving it 


# Running

```
bash run.sh [path/to/ply/file] [experiment name]

For example:
>>> bash run.sh ./logs/test/data.ply test
```


