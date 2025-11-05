# Hand2Sim - A tool for retargeting hand interaction trajectories to robots across simulators

> Authors: [Sriram Krishna](https://sriramsk1999.github.io/), [Homanga Bharadhwaj](https://homangab.github.io/)

This project aims to be a real2sim pipeline which takes in an in-the-wild video of a human performing an action, extracts the hand trajectory, retargets it to a robot arm, and replays it in simulation.

![Demo1](assets/robohive-pjaw-ZY20210800001_H1_C13_N46_S238_s03_T4.gif)

![Demo2](assets/isaacsim-allegro-ZY20210800003_H3_C14_N42_S207_s05_T2.gif)

> **Note**: This is a work-in-progress building a system for hand2robot retargeting. Lots of features to implement!

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
python hand2sim.py --dataset "hoi4d" --path "/path/to/hoi4d/video/" --env_name isaacsim --embodiment allegro
```

Please refer `hand2sim.py` for other options (using different datasets/environments/embodiments).

**NOTE:** Might need to set `MESA_GL_VERSION_OVERRIDE=4.6` to make it work on a headless system.

## Roadmap

- [x] Add HOI4D dataset support
- [x] Support MuJoCo (RoboHive) and IsaacSim environments
- [x] Add parallel jaw Franka and Franka with Allegro hand
- [x] Initial support for generic dataset (HAMER, MONST3R, ContactHands)
- [ ] Refactor generic dataset to use [HAPTIC](https://judyye.github.io/haptic-www/) and [AnyTeleop](https://yzqin.github.io/anyteleop/)
- [ ] Add more arms/hands - xArm, UR5, Leap Hand, Shadow Hand
- [ ] Add support for ManiSkill

The eventual goal for this project is to be able take in arbitrary hand-object trajectories and retarget them across simulators/embodiments to serve as a general-purpose toolkit for real2sim manipulation.

## Acknowledgements

This codebase makes use of [AnyTeleop](https://yzqin.github.io/anyteleop/). We thank the authors for releasing their work!
