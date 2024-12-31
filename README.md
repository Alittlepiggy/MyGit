
#### Install

```
# 1 创建环境
conda create -n opencd python=3.8
conda activate opencd

# 2 安装torch
# 方式1：
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
# 方式2：
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113

# 3 验证torch安装是否为gpu版
import torch
print(torch.__version__)  # 打印torch版本
print(torch.cuda.is_available())  # True即为成功
print(torch.version.cuda)
print(torch.backends.cudnn.version())

# 4 安装其他依赖库
git clone https://github.com/Alittlepiggy/MyGit.git
cd ./MyGit

# 4.1 安装 OpenMMLab 相关工具
pip install -U openmim
mim install mmengine
mim install "mmcv==2.0.0"
mim install "mmpretrain>=1.0.0rc7"  # (本地安装版本为1.2.0)
pip install "mmsegmentation==1.2.2"
pip install "mmdet==3.0.0"

# 4.2 编译安装open-cd
pip install -v -e .

# 5 可能缺少的库
pip install ftfy
pip install regex
```


#### Train
```
python tools/train.py configs/changer/changer_ex_r18_512x512_40k_levircd.py --work-dir ./changer_r18_levir_workdir
```

#### Test
```
# get .png results
python tools/test.py configs/changer/changer_ex_r18_512x512_40k_levircd.py changer_r18_levir_workdir/latest.pth --show-dir tmp_infer
# get metrics
python tools/test.py configs/changer/changer_ex_r18_512x512_40k_levircd.py changer_r18_levir_workdir/latest.pth
```

#### Your use
```
# 运行顺序，需要修改路径
python duiqi.py #对齐操作
python fenge.py  #分割图片成1024*1024的图片
python try_my_change_detection.py #推理
python yunsuanlvbo.py #图像处理
python find_the_pos.py #输出处理，在原图像上标注变化区域
```

