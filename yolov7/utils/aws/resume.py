# ==============================================================
# utils/aws/resume.py － 在 AWS 等雲端主機重開機後，自動接續所有中斷的訓練
# 會找出目前資料夾底下所有 last.pt，逐一用 --resume 在背景重新啟動
# ==============================================================
# Resume all interrupted trainings in yolor/ dir including DDP trainings
# Usage: $ python utils/aws/resume.py

import os
import sys
from pathlib import Path

import torch
import yaml

sys.path.append('./')  # to run '$ python *.py' files in subdirectories

port = 0  # --master_port
path = Path('').resolve()
# 找出所有 last.pt，依照當初的設定（單 GPU 或 DDP）重新啟動訓練
for last in path.rglob('*/**/last.pt'):
    ckpt = torch.load(last, weights_only=False)
    if ckpt['optimizer'] is None:
        continue

    # Load opt.yaml
    with open(last.parent.parent / 'opt.yaml') as f:
        opt = yaml.load(f, Loader=yaml.SafeLoader)

    # Get device count
    d = opt['device'].split(',')  # devices
    nd = len(d)  # number of devices
    ddp = nd > 1 or (nd == 0 and torch.cuda.device_count() > 1)  # distributed data parallel

    if ddp:  # multi-GPU
        port += 1
        cmd = f'python -m torch.distributed.launch --nproc_per_node {nd} --master_port {port} train.py --resume {last}'
    else:  # single-GPU
        cmd = f'python train.py --resume {last}'

    cmd += ' > /dev/null 2>&1 &'  # redirect output to dev/null and run in daemon thread
    print(cmd)
    os.system(cmd)
