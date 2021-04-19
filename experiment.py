from pathlib import Path
import os
import subprocess
import sys
import random

PROTOYPE_NAME:str = "0_Pipeline" 
EXPERIMENT_NAME:str = "Jump18"

ENV_PATH = rf"C:\Users\user\Desktop\RLNav\NavigationEnvironments\P0\{EXPERIMENT_NAME}\Env.exe"

for i in range(10):
    for batch_size in [256, 512, 1024, 2048, 4096]:
        TREATMENT:str = f"PPO_{batch_size}"
        run_id = Path(".") / PROTOYPE_NAME / EXPERIMENT_NAME / TREATMENT
        experiment_path = "Results" / run_id
        experiment_path.mkdir(parents=True, exist_ok=True)
        RUN_COUNT = len(next(os.walk(experiment_path))[1])
        run_id /= f"Run_{RUN_COUNT}"

        subprocess.run(["mlagents-learn", f"rlnav/configs/{TREATMENT}.yaml", f"--env={ENV_PATH}", 
        f"--run-id={run_id}", "--time-scale=20", f"--base-port={i + 6000 + random.randint(0,100) * 10}" , "--width=480", "--height=480",
        "--torch-device=cuda"])

#subprocess.run(["mlagents-learn", "rlnav/configs/PPO.yaml", f"--env={ENV_PATH}", f"--run-id={run_id}"])



