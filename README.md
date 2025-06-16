# Retargeting human videos to robot trajectories

This project aims to be a real2sim pipeline which takes in an in-the-wild video of a human performing an action, extracts the hand trajectory, retargets it to a robot arm, and replays it in simulation.

## Installation

``` sh
pixi install
pixi run install-mano
pixi shell
robohive_init
```

Download `MANO_RIGHT.pkl` from the [MANO website](https://mano.is.tue.mpg.de/) and place it at `mano/models/`.

## Usage

Example usage:

``` sh
python real2sim.py --dataset "hoi4d" --path "/path/to/hoi4d/video/" --env_name isaacsim --embodiment allegro
```

Please refer `real2sim.py` for other options (using different datasets/environments/embodiments).

**NOTE:** Might need to set `MESA_GL_VERSION_OVERRIDE=4.6` to make it work on a headless system.


## Acknowledgements

This codebase makes use of [AnyTeleop](https://yzqin.github.io/anyteleop/). We thank the authors for releasing their work!
