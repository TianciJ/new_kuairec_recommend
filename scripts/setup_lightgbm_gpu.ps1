param(
    [string]$Python = 'D:\Anaconda\envs\recommend\python.exe',
    [string]$Conda = 'D:\Anaconda\Scripts\conda.exe'
)
$ErrorActionPreference = 'Stop'
$prefix = & $Python -c 'import sys; print(sys.prefix)'
if ($LASTEXITCODE -ne 0) { throw '无法读取 Python 环境' }
# 此构建包含 OpenCL；固定 Boost 版本，避免新版离线 kernel cache 的兼容问题。
& $Conda install -p $prefix --override-channels -c conda-forge 'liblightgbm=4.6.0=cpu_hb4e115d_0' 'libboost=1.84' -y
if ($LASTEXITCODE -ne 0) { throw 'OpenCL 原生库安装失败' }
& $Python -m pip install 'lightgbm==4.6.0' 'scikit-learn>=1.6,<2'
if ($LASTEXITCODE -ne 0) { throw 'Python 包安装失败' }
@'
import importlib.util, os, shutil, sys
from pathlib import Path
prefix = Path(sys.prefix)
package = Path(importlib.util.find_spec('lightgbm').origin).parent
targets = list(package.rglob('lib_lightgbm.dll'))
if len(targets) != 1:
    raise RuntimeError('找不到唯一的 LightGBM 原生库')
target = targets[0]
backup = target.with_name('lib_lightgbm.original.dll')
if not backup.exists():
    shutil.copy2(target, backup)
shutil.copy2(prefix / 'Library/bin/lib_lightgbm.dll', target)
handle = os.add_dll_directory(str(prefix / 'Library/bin'))
import lightgbm as lgb
import numpy as np
rng = np.random.default_rng(42)
x = rng.normal(size=(4096, 12)).astype('float32')
y = (x[:, 0] + x[:, 1] > 0).astype('int8')
model = lgb.train({'objective': 'binary', 'device_type': 'gpu', 'gpu_platform_id': 0,
                   'gpu_device_id': 0, 'max_bin': 63, 'num_leaves': 15, 'num_threads': 4},
                  lgb.Dataset(x, label=y), num_boost_round=3)
assert np.isfinite(model.predict(x[:3])).all()
print('GPU smoke passed; verify the GPU name in the LightGBM log above.')
'@ | & $Python -
if ($LASTEXITCODE -ne 0) { throw 'GPU 实测失败，不回退 CPU' }
