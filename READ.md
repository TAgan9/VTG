# MS-DETR
1. 项目介绍
2. 项目安装
创建环境
```bash
conda create -n vtg python=3.7 -y
# 安装项目依赖项
pip install -U pip wheel setuptools==59.5.0
pip install numpy scipy tqdm tensorboard scikit-learn matplotlib ftfy regex pandas tabulate
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html

# 验证安装
python -c "import torch; import numpy as np; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# 还需要取detr项目中copy其util文件夹到ms_detr文件夹中
```
QVHightlight训练脚本
```bash
bash ms_detr/scripts/train.sh
# 可视化
tensorboard --logdir=/srv/home/ganxinchao/MS-DETR/results/hl-video_tef-exp-2025_11_10_13_41_59/tensorboard_log --port=6006
```
QVHighlight测试脚本
```bash
bash ms_detr/scripts/inference.sh results/{direc}/model_best.ckpt 'val'
bash ms_detr/scripts/inference.sh results/{direc}/model_best.ckpt 'test'
```