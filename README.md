# Retargeting human web videos to robot trajectories

## Installation

``` sh
conda env create --file environment.yml
conda activate real2sim
robohive_init
```

Download `MANO_RIGHT.pkl` from the [MANO website](https://mano.is.tue.mpg.de/) and place it at `mano/models/`.

## Usage

Example usage:

``` sh
python real2sim.py --dataset "hoi4d" --path "/path/to/hoi4d/video/" --env_name isaacsim --embodiment allegro
```

Please refer `real2sim.py` for other options (using different datasets/environments/embodiments).

## Pipeline (WIP)

The goal is to be able to take in any video of a human performing an action, extract the hand trajectory, retarget it to a Franka arm, and replay it in RoboHive.

To set up this entire pipeline, we require the following annotations:

- **Camera Pose** - To compensate for camera motion when replaying the trajectory
- **Hand Pose** - To transfer hand motion to the end effector
- **Hand-Object Contact Flag** - To get open/close actions for the end effector.

These annotations are acquired with the following off-the-shelf models:

- **Camera Pose** - Monst3R
- **Hand Pose** - HaMeR
- **Depth** - Monst3R (HaMeR provides weak perspective hand positions, can be inconsistent over a video. Thus we can use a dedicated depth estimator and then only use hand orientation from HaMeR, not the hand position itself).
- **Hand-Object Contact Flag** - ContactHands

