from setuptools import setup
from pathlib import Path
import sys
import subprocess

if __name__ == "__main__":
    # print("Installing rlnav will attempt to install ml-agents and stable-baselines in an editable form.")

    current_dir = Path().absolute()
    parent_dir = current_dir.parent.absolute()

    # editable_requirements = [f'stable-baselines3 @ file://{str(parent_dir / Path("stable-baselines3/"))}',
    #                          f'mlagents-envs @ file://{str(parent_dir / Path("ml-agents/ml-agents-envs/"))}',
    #                          f'gym-unity @ file://{str(parent_dir / Path("ml-agents/gym-unity/"))}',
    #                          f'mlagents @ file://{str(parent_dir / Path("ml-agents/ml-agents/"))}' ]

    setup(install_requires=[])
