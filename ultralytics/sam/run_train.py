"""Auto-restart training wrapper. Restarts until all epochs done."""

import subprocess, sys, os, torch

MAX_RETRIES = 30
PYTHON = sys.executable
CKPT_PATH = r'E:\sam\output_xutong\best\checkpoint.pt'
TARGET_EPOCHS = 50  # Must match config.NUM_EPOCHS


def get_epoch():
    if os.path.isfile(CKPT_PATH):
        try:
            state = torch.load(CKPT_PATH, map_location='cpu')
            return state.get('epoch', 0)
        except Exception:
            return 0
    return 0


for attempt in range(1, MAX_RETRIES + 1):
    current = get_epoch()
    print(f'\n{"="*50}', flush=True)
    print(f'  Attempt {attempt} | Current epoch: {current}/{TARGET_EPOCHS}', flush=True)
    print(f'{"="*50}\n', flush=True)

    if current >= TARGET_EPOCHS:
        print(f'Training complete ({current} epochs).', flush=True)
        break

    result = subprocess.run([PYTHON, '-m', 'sam2_lora.train'], cwd=r'E:\sam')

    new_epoch = get_epoch()
    if new_epoch >= TARGET_EPOCHS:
        print(f'\nAll {TARGET_EPOCHS} epochs completed!', flush=True)
        break
    elif new_epoch > current:
        print(f'Progress: {current} -> {new_epoch} epochs. Restarting...', flush=True)
    else:
        print(f'No progress. Exit code: {result.returncode}', flush=True)
        if result.returncode != 0 and new_epoch == 0:
            print(f'Fatal error with no checkpoint. Stopping.', flush=True)
            break
