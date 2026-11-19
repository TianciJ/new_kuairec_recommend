# KuaiRec 推荐系统

从零搭建基于 KuaiRec 的短视频推荐项目。当前已完成官方原始数据的下载、校验和整理。

## 数据

数据来自 [Zenodo 18164998](https://zenodo.org/records/18164998)（2026-01-06 发布），已下载并通过 MD5 校验。
官方下载版本的来源和校验值保存在 `data/sources/kuairec.json`；如需重新获取，可通过上述官方发布页下载。本地 `big_matrix.csv` 已清理日期不一致的记录，`item_daily_features.csv` 已精简字段，`video_raw_categories_multi.csv` 已去除完全重复记录；当前校验值见 `data/README.md`。

当前保留的核心文件位于 `data/raw/KuaiRec 2.0/data/`（保留官方压缩包内的目录名）：

| 文件 | 用途 |
| --- | --- |
| `big_matrix.csv` | 较大规模的用户视频交互记录 |
| `item_daily_features.csv` | 视频按日统计特征 |
| `item_categories.csv` | 视频类别 |
| `social_network.csv` | 用户社交关系 |
| `kuairec_caption_category.csv` | 视频文本与层级类别 |

用户原始特征 `user_features_raw.csv` 和视频原始分类 `video_raw_categories_multi.csv` 保存在 `data/raw/supplemental/`。
当前保留 7 份数据源 CSV，可用于基线模型及后续特征建模。数据文件均已加入 `.gitignore`。

后续推荐实验只使用 `big_matrix.csv` 作为交互数据源，不使用 `small_matrix.csv`。
已从 `big_matrix.csv` 删除日期不一致的 15,530 条记录，再按原始视频分类表的 ID 范围删除 26,567 条交互，当前剩余 12,488,709 条。保留行的原始内容及顺序不变。

`item_daily_features.csv` 已按用户要求删除 28 列，并同步过滤视频 ID，当前保留 30 列、342,281 行，不保留该表原版备份；处理记录见 `data/README.md`。

`video_raw_categories_multi.csv` 已去除 55 条完全重复记录，每组保留第一次出现的一条；当前为 26,771 行、13 列，仍覆盖 10,683 个视频，未保留原版备份。

所有含 `video_id` 的现有数据表（包括 train/test、视频类别和文本表）已统一过滤为原始视频分类表覆盖的 10,683 个视频范围，移除了其未覆盖的 45 个视频。保留原始 ID，不重新编号；具体记录数、删除 ID 和校验值见 `data/processed/video_id_filter_metadata.json`。

## 后续实现顺序

1. 检查 `big_matrix.csv` 的字段、缺失值、重复交互与 ID 范围。
2. 明确推荐目标与正反馈定义；基于已固定的训练与测试划分建立评估流程，按需从训练集末尾留出验证集。
3. 跑通热门推荐和协同过滤基线，建立 Recall@K、NDCG@K 等评估流程。
4. 再实现召回、排序及演示接口。

训练与测试数据已从清理后的 `big_matrix.csv` 按北京时间 `2020-08-29 00:00:00` 划分，输出位于 `data/processed/`：

- `train.csv`：分界之前的 9,923,901 条交互，占 79.4630%。
- `test.csv`：分界及之后的 2,564,808 条交互，占 20.5370%。
- `split_metadata.json`：固定边界、记录数、时间范围和文件校验值。

视频范围过滤脚本为 `scripts/filter_video_ids.py`，详细规则见 `data/README.md`。训练与测试保留全部 8 个字段，时间边界不变，未额外筛选正反馈或冷启动样本。
视频按日统计特征需要按预测时点选择，避免使用未来信息。

## 召回基线

`recall/` 下的热门和 ItemCF 基线依赖 `numpy`、`pandas`、`scipy`。两个实现直接读取 CSV，不依赖 `data/sources/kuairec.json` 或其他 JSON，不执行哈希校验，也不读取测试集。第三路 gSASRec 另需 PyTorch，见下节。

| 文件 / 类 | 默认规则 |
| --- | --- |
| `recall/hot_recall.py` / `HotRecall` | 累加 2020-08-22 至 2020-08-28 的每日 `play_user_num`，按分数降序召回。同分按视频 ID 升序；该分数是累计人日，不是 7 天去重人数。 |
| `recall/itemcf_recall.py` / `ItemCFRecall` | 截断前 `play_duration / video_duration >= 1` 为正反馈；用户-视频只计一次，取最近一次正反馈时间。默认每用户保留最近 200 个正反馈视频计算共现余弦相似度，每视频保留 100 个相似邻居。 |

两路均根据完整训练期观看历史排除已看视频，未完播的观看也会被排除。ItemCF 的 200 条上限只用于相似度和兴趣分数计算，不截断已看过滤集合；可用 `--max-history 0` 取消上限，但计算成本会增加。候选分数分别是播放人数累计值和相似度累加值，不是概率，不能直接按原始分数跨路比较。

热门召回无需模型训练；ItemCF 的 `fit()` 是离线统计和构建相似度索引，无梯度优化。新用户或没有正反馈的用户在 ItemCF 中返回空列表，可由热门召回兜底。公开状态、广告等业务过滤可通过 `recommend(..., candidate_ids=...)` 传入允许的候选 ID；两个基线不自动实现业务过滤。

在项目根目录运行以下命令，为用户 0 生成前 20 个候选：

```powershell
python recall/hot_recall.py --user-id 0 --top-k 20
python recall/itemcf_recall.py --user-id 0 --top-k 20
```

`--cutoff` 默认北京时间 `2020-08-29`；热门召回按完整自然日聚合，要求截止时刻是零点，`--window-days` 默认 7。ItemCF 可通过 `--positive-threshold` 和 `--similarity-top-k` 调整正反馈阈值及相似邻居数。两者均只纳入 `timestamp < cutoff` 的观看历史。

也可以通过 `--save-index 路径.npz` 保存离线结果，通过 `--load-index 路径.npz` 直接加载；加载时使用索引里保存的参数，不再读取原始 CSV 或重新拟合。保存索引时拒绝覆盖已有文件。用 `--output 路径.csv` 可保存该次召回结果。

已用真实训练集生成以下索引，可以直接加载：

```powershell
python recall/hot_recall.py --load-index artifacts/recall/hot.npz --user-id 0 --top-k 20
python recall/itemcf_recall.py --load-index artifacts/recall/itemcf.npz --user-id 0 --top-k 20
```

Python 调用示例：

```python
from recall.hot_recall import HotRecall
from recall.itemcf_recall import ItemCFRecall

hot = HotRecall.load("artifacts/recall/hot.npz")
itemcf = ItemCFRecall.load("artifacts/recall/itemcf.npz")
hot_candidates = hot.recommend(user_id=0, top_k=100)
itemcf_candidates = itemcf.recommend(user_id=0, top_k=100)
# 返回值均为 [(video_id, score), ...]；同一实例可服务多个用户。
```

验证已覆盖手算相似度、时间边界、重复观看、历史上限、完整已看过滤、新用户、无正反馈、空输入、候选限制、索引保存和加载。真实数据上两路均已完成离线构建与测试评估，结果见下节的三路对照。

## 第三路：gSASRec

实现位于 `recall/gsasrec_recall.py`，类名 `GSASRecRecall`。参考
[gSASRec 论文（RecSys 2023）](https://arxiv.org/abs/2308.07192)及
[作者 PyTorch 实现](https://github.com/asash/gSASRec-pytorch)，为本项目独立适配实现，
不是论文数据、超参数和评估协议的逐项复现。TIGER 不在本项目中实现。

### 数据与训练规则

- 只读取 `train.csv` 中北京时间 `2020-08-29 00:00:00` 之前的交互。
- 正反馈定义：`play_duration >= video_duration`。按用户和时间稳定排序，保留不同时间的重复观看，删除完全重复事件。
- 每用户使用最后 `max_length + 1` 个正反馈事件构建错位输入和标签；默认最近 101 个事件，最多 100 个训练位置。不是遍历所有历史窗口。
- 原始视频 ID 映射至内部 `1..N`，内部 `0` 专用于 padding；返回结果还原原始 ID。映射只基于相应训练期的正反馈视频。
- 两层因果自注意力、可学习位置向量、独立输入/输出视频 embedding。左填充位置不参与损失；不能读取未来位置。
- 每个有效位置均匀、有放回抽取 256 个负样本，排除当前位置的真实目标；不排除其他历史视频，不将采样负例解释为明确不喜欢。
- 使用稳定的 gBCE：`[beta * softplus(-positive_logit) + sum(softplus(negative_logits))] / (k+1)`，在有效位置取平均。`alpha=k/(N-1)`，`beta=1-t+t*alpha`，默认 `t=0.75`。不是普通 BCE 冒充 gBCE。
- 保持磁盘 train/test 不变。先把 train 最后两天（8 月 27、28 日）用作内部时间验证；此前的数据训练模型，以全候选 Recall@200（逐用户平均）选训练轮数。
- 验证正例按用户-视频去重，保留已看正例与冷启动视频在分母中；没有正例的用户不计入平均，无法产生候选的用户记为零命中。
- 选好轮数后，从相同种子重新初始化，用完整 train 训练该轮数。test 不参与早停、选参或训练；只有显式指定 `--test` 才进行最终测试。

### 本机 GPU 运行

已检测本机 `D:/Anaconda/envs/recommend/python.exe` 可用，环境为 Python 3.12、
PyTorch 2.11.0+cu128、NumPy 2.4.6、pandas 3.0.3、SciPy 1.17.1。
GPU 是 RTX 3050 Ti Laptop 4GB。Anaconda base 环境未安装 torch，不能直接用 base 的 `python`。

默认参数为序列长 100、向量维度 64、2 层、1 个头、dropout 0.2、batch size 32、
256 个负样本、Adam 学习率 0.001、最多 20 轮、验证连续 4 轮不提升早停。
这是 4GB 显存的起步配置，不声称最优。CLI 默认 `--device cuda`；CUDA 不可用时报错，
不会静默在 CPU 上长时间训练。固定随机种子 42，不保证跨硬件/库版本逐位相同。

```powershell
# 首次训练并保存：拒绝覆盖已有 checkpoint
& 'D:/Anaconda/envs/recommend/python.exe' recall/gsasrec_recall.py --device cuda --save-index artifacts/recall/gsasrec.pt --top-k 20

# 加载第三路，给用户 0 召回 200 个候选，不重新训练
& 'D:/Anaconda/envs/recommend/python.exe' recall/gsasrec_recall.py --load-index artifacts/recall/gsasrec.pt --user-id 0 --top-k 200

# 显式运行一次最终测试；训练期已看过滤与前两路一致
& 'D:/Anaconda/envs/recommend/python.exe' recall/gsasrec_recall.py --load-index artifacts/recall/gsasrec.pt --test data/processed/test.csv --top-k 200

# 小数据公式/遮罩/采样/时间边界/过滤/保存加载/训练流程测试
& 'D:/Anaconda/envs/recommend/python.exe' -m unittest discover -s tests -v
```

内存不足可降低 `--batch-size`、`--max-length` 或 `--negatives`，更改参数需要重新训练。
`--load-index` 使用文件内配置；训练参数不会覆盖已存模型。`--output` 可另存某用户的召回 CSV。

```python
from recall.gsasrec_recall import GSASRecRecall

gsasrec = GSASRecRecall.load("artifacts/recall/gsasrec.pt", device="cuda")
third_route = gsasrec.recommend(user_id=0, top_k=200)
# 与前两路相同：[(原始 video_id, score), ...]
# candidate_ids / exclude_ids 可传允许集合和额外排除集合。
```

模型会过滤用户完整训练期已看视频（含未完播观看），按分数降序、视频 ID 升序返回；
没有正反馈历史的新用户返回空列表，由热门召回另行兜底。没有训练正反馈的视频没有可学习的
候选向量，不作为 gSASRec 候选；测试期新视频仍保留在评估分母。
分数是未归一化的点积，可以为负，不是观看概率，也不能与其他两路的原始分数直接相加。
三路合并时按每位用户的 `video_id` 去重；这里保留独立召回接口，不擅自设定融合权重。

`gsasrec.pt` 包含网络权重、超参数、原始视频 ID 映射、每用户最近正反馈序列、完整已看稀疏记录，
以及内部验证/最终重训日志。不是哈希文件，不依赖旁路 JSON；加载使用 `weights_only=True`。
该文件保存的是推理 checkpoint，不包含优化器状态，不支持中断处精确续训。

### 本次训练与检查结果（2026-09-23）

在本机 RTX 3050 Ti 上完成内部验证训练：第 3 轮的验证 Recall@200 最优（2.9464%），
第 7 轮后触发早停。随后从头在完整 9,923,901 条 train 交互上重训 3 轮，
保存 `artifacts/recall/gsasrec.pt`（78,144,689 字节）。模型候选词表包含 8,824 个训练正反馈视频。
验证期与测试期不相同，验证分数不能与测试分数直接比较。

固定测试期为 2020-08-29 至 2020-09-05。正例为播放时长不小于视频时长的交互，
按用户-视频去重；对有正例的 6,930 位用户计算各自 Recall 后取平均。
707,227 个测试正例均保留在分母，包括训练期已看视频和冷启动视频。
模型一律过滤训练期全部已看视频；测试期历史不滚动更新，不做采样候选评估。

| 独立召回方法 | 用户平均 Recall@200 | 命中的用户-视频正例 |
| --- | ---: | ---: |
| 热门 | 9.0880% | 73,539 |
| ItemCF | 9.3255% | 81,372 |
| gSASRec | 10.6239% | 83,661 |

gSASRec 的 micro Recall 为 11.8294%，与上表用户平均口径不同，不混用。
这是一次固定种子的基线结果，没有使用 test 选轮数或调参，也未进行显著性检验。
7 项单元测试通过；全量 7,176 位有序列的用户均返回 200 个候选，检查无重复、无已看，
CPU/GPU 加载推理抽查排名一致、分数在浮点容差内一致。

### 延长训练：最多 500 轮，耐心值 20

保持其余超参数和验证口径不变。连续 20 轮未刷新内部验证 Recall@200 的全局最佳值时停止，
而不是与上一轮比较；同分也计入未提升。随后仍以最佳轮数在完整 train 上从头重训。
原推理 checkpoint 不含优化器状态，因此此实验从固定种子重新开始，不是精确断点续训。
原 `gsasrec.pt` 保留，新模型和日志独立保存：

```powershell
& 'D:/Anaconda/envs/recommend/python.exe' -u recall/gsasrec_recall.py --device cuda --epochs 500 --patience 20 --save-index artifacts/recall/gsasrec_500_patience20.pt --test data/processed/test.csv --top-k 200
```

本次运行日志：`artifacts/recall/gsasrec_500_patience20.log`。
命令末尾的测试仅在完整训练结束后执行一次，不用于挑选轮数。

本次已完成：第 52 轮取得最佳内部验证 Recall@200 **4.224841%**，
第 53～72 轮连续 20 轮没有刷新最佳值，因此在第 72 轮停止选择阶段，未跑满 500 轮。
随后使用完整 train 从头重训 52 轮，保存上述新 checkpoint，未覆盖原 3 轮模型。

同一固定测试集、同一逐用户平均口径：

| gSASRec 版本 | 完整 train 重训轮数 | 测试 Recall@200 | 命中用户-视频正例 |
| --- | ---: | ---: | ---: |
| 原 `gsasrec.pt` | 3 | 10.623882% | 83,661 |
| 新 `gsasrec_500_patience20.pt` | 52 | **19.712620%** | **130,809** |

两次均评估 6,930 位有正例用户、707,227 个正例；新模型每用户平均返回 200 个候选。
新模型 micro Recall 为 18.496042%，不要与表中的用户平均 Recall 混淆。
本轮测试只在选轮数和重训结束后运行；结果提高约 9.089 个百分点。

使用本轮训练好的第三路模型：

```powershell
& 'D:/Anaconda/envs/recommend/python.exe' recall/gsasrec_recall.py --load-index artifacts/recall/gsasrec_500_patience20.pt --user-id 0 --top-k 200
```

## LightGBM 粗排

`ranking/` 在三路各 200 个候选合并去重后，用完播二分类分数保留前 200 个。模型、历史召回快照、样本缓存、评估结果和日志存放在 `artifacts/ranking/lightgbm/`，不会修改原始数据或已有召回 checkpoint。

候选与样本缓存按时间窗口、候选数和模型配置复用，不做文件 hash、大小或修改时间校验。更换数据或上游模型时，使用新的 `--output` 目录。

- `ranking/features.py`：按截止时间构造 88 个数值特征，包括三路分数/排名/命中标记、用户最近 200 次观看偏好、准确的最近 1/7 天行为计数、视频过去 1/7 天统计、31 个多类别标记、用户与类别/作者/时长的匹配特征。时长从毫秒转换成秒；观看比例均值先截到 1。多类别不展开训练行。
- `ranking/candidates.py`：复用现有召回接口，检查模型截止时间，缓存各时间截点的候选。没有进入某路 Top-200 的分数和排名记为缺失，不伪造为零；gSASRec 负分仍保留。
- `ranking/lightgbm_ranker.py`：训练、保存加载、批量评分和评估。正式训练固定 `device_type=gpu`，GPU 不可用时报错，不自动改成 CPU。

### 样本和时间口径

正例为 `play_duration >= video_duration`。每个截点先产生候选和特征，再用后续窗口的首次已观测观看贴标签。已观看未完播为 0；没有记录的候选保持未知，不加入二分类训练。模型学到的是已记录观看条件下的完播规律，不把输出宣称为经过校准的全量候选完播概率。

实际训练日志只有 7 月 5—12 日、8 月 1—10 日、8 月 27—28 日有交互，不能假设每个自然日都有数据。因此使用以下固定窗口：

| 阶段 | 特征与召回截止时间（北京零点） | 标签窗口（左闭右开） |
| --- | --- | --- |
| 粗排训练 1 | 2020-08-03 | 08-03 至 08-06 |
| 粗排训练 2 | 2020-08-06 | 08-06 至 08-09 |
| 粗排训练 3 | 2020-08-09 | 08-09 至 08-11 |
| 验证 | 2020-08-27 | 08-27 至 08-29 |
| 最终测试 | 2020-08-29 | 08-29 至 09-06 |

用户长期偏好使用截点前最近 200 次观看，避免日期空档使 7 天偏好全部清空。准确的 1/7 天观看次数仍作为独立特征。视频日统计只取截止日期之前的数据；当前观看时长和观看比例不进入输入。`user_features_raw.csv`、社交关系快照和文本不用于第一版，以免无法确认历史可用时间的动态属性泄漏。

历史 gSASRec 最多训练 100 轮、耐心值 10，使用截止前最后两个有交互的自然日作内部验证，再按选出的轮数重训。8 月 29 日的最终召回复用已有 `gsasrec_500_patience20.pt`。LightGBM 默认最多 1,000 轮，以验证 Logloss 早停（耐心值 50），随后用三个训练窗口和验证窗口的有标签样本按固定轮数重训。测试集不参与模型选择。

分类评估基于首次观看标签；推荐评估沿用原召回协议：窗口内任意一次完播为正例，按用户视频去重，分母保留训练期已看和冷启动正例，测试历史不滚动更新。报告 LightGBM、同候选条件的 gSASRec、RRF 排名融合（常数 60）的 Recall@200、NDCG@200、候选正例保留率和打分耗时。候选只包含截点前可确认已发布且有历史视频日元数据的物品，因此这里的 gSASRec 数字可能与未加该限制的旧实验有差异。

### 本机 GPU 运行

本机 Windows/RTX 3050 Ti 使用 OpenCL GPU 后端。可复现环境固定为 LightGBM 4.6.0、conda-forge 的 `liblightgbm=4.6.0=cpu_hb4e115d_0`、Boost 1.84；发行包名中的 `cpu` 也包含 OpenCL GPU 支持。普通 PyPI DLL 在本机未启用 GPU，安装脚本会备份后替换为同版本 OpenCL DLL，并运行 GPU 检查。训练模块会先注册该环境的 DLL 搜索目录。

```powershell
# 新环境首次配置；当前本机环境已配置并实测 GPU。
./scripts/setup_lightgbm_gpu.ps1

# 一次完成历史快照、特征、GPU 训练、验证和最终测试。
& 'D:/Anaconda/envs/recommend/python.exe' -u -m ranking.lightgbm_ranker --stage all

# 已有缓存可继续；缓存配置或源文件不匹配时会报错，需使用新的 --output 目录。
& 'D:/Anaconda/envs/recommend/python.exe' -m ranking.lightgbm_ranker --stage recommend --user-id 0 --top-k 200

# 时间边界、标签、候选、评估与保存加载检查。
& 'D:/Anaconda/envs/recommend/python.exe' -m unittest discover -s tests -v
```

使用新候选数量或训练超参数时，应指定独立的 `--output`。候选按批生成和评分，有标签样本才展开成训练矩阵；正式 GPU 训练日志应显示 `Using GPU Device: NVIDIA GeForce RTX 3050 Ti Laptop GPU`。OpenCL 负责直方图计算，数据处理和部分训练步骤仍在 CPU。

输出包括 `model.txt`、`model_meta.json`、`feature_importance.csv`、`validation_metrics.json`、`test_metrics.json` 和示例用户推荐 CSV。`LightGBMCoarseRanker.load(...)` 可加载模型，`predict(features)` 批量评分，`recommend(user_id, candidates, feature_store)` 对给定候选排序；特征名称和顺序必须与训练一致。

### 首次完整 GPU 训练结果

本机已完成四个历史 gSASRec 快照，其完整历史重训轮数依次为 42、37、38、40。三个粗排训练窗口共 1,300,083 条有标签样本，验证窗口有 116,811 条。LightGBM 在 RTX 3050 Ti 上以验证 Logloss 选出 **63 轮**，验证 AUC **0.769118**、Logloss **0.527439**；最终用合计 **1,416,894** 条样本固定重训 63 轮，正例占比 41.5952%。输入为 88 列，LightGBM 实际使用其中 85 列。

以下 Recall 均为逐用户平均，NDCG 使用二元完播相关性。三种方法从同一份三路候选中各保留 200 个：

| 方法 | 验证 Recall@200 | 测试 Recall@200 | 测试 NDCG@200 | 测试命中正例 |
| --- | ---: | ---: | ---: | ---: |
| LightGBM 完播粗排 | 3.5043% | 11.5139% | 0.091833 | 87,518 |
| gSASRec 原排序 | 4.1549% | **19.7126%** | **0.186198** | **130,809** |
| RRF 三路融合 | **5.3464%** | 16.3498% | 0.160958 | 117,314 |

测试包含 6,930 位有正例用户、707,227 个用户-视频正例；每位评估用户均返回 200 个视频。三路候选合并后共 3,819,426 行，候选池宏平均 Recall 为 28.9829%。LightGBM 保留了候选池中 40.6729% 的正例，而 gSASRec 保留了 60.7919%。完整结果见 `artifacts/ranking/lightgbm/test_metrics.json`。

**首版已跑通，但排序效果低于现有 gSASRec，不能作为效果提升来替换基线。** 完播分类 AUC 与全候选 Recall 衡量的目标不同，且训练仅覆盖有观看记录的候选。特征重要性中视频时长占总增益约 66.11%，提示模型可能偏重“容易完播”的视频；这只是待验证的解释，尚未通过消融实验确认。后续应在验证集上研究目标与采样偏差，不能反复用本次测试集选择方案。

LightGBM 选轮数及验证分类预测共约 8.01 秒，最终 GPU 重训约 5.42 秒；这些时间不包含历史 gSASRec 训练和样本构建。测试的全量特征转换约 7.06 秒、批量模型预测约 3.04 秒，包含读取、建特征快照和召回的完整测试流程约 129.98 秒。这些是本机离线批量时间，不代表线上单请求延迟。

7 项单元测试通过；保存后重新加载模型，普通用户 `0` 的前 20 个推荐与全量评估导出结果一致且不含已看视频，新用户 `999999` 的热门候选兜底也返回 20 个不重复、有序的结果。核验记录保存在 `artifacts/ranking/lightgbm/verification.json`。

## DIN 精排

新增链路为 **三路召回合并 → LightGBM 保留 200 个 → DIN 输出前 20 个**。默认继续预测完播，
沿用 `play_duration >= video_duration` 的正例定义。DIN 分数反映已观测观看样本中的完播倾向，
不视为经过校准的全候选概率，也不与召回原始分数相加。

- `ranking/din_ranker.py`：原生 PyTorch DIN、批量评分、级联推荐及 checkpoint 保存加载。
- `ranking/din_data.py`：截止前最近 50 次完播序列、ID 词表和数值预处理。
- `ranking/din_pipeline.py`：复用现有历史候选与特征缓存，完成训练、推荐、最终测试。
- `tests/test_din_ranker.py`：时间隔离、注意力梯度、空历史、未知 ID、训练和序列化检查。

### 模型与时间口径

候选视频与历史视频共享 32 维 embedding。局部激活网络输入候选、历史、两者之差及逐元素乘积，
输出不经 softmax 归一化的权重，对历史向量加权求和；padding 权重为零。
拼接用户向量、候选向量、兴趣向量和原有 88 个数值特征，经 MLP 输出完播 logit。
使用 BCEWithLogitsLoss、AdamW、梯度裁剪及 dropout。此版本保留
[DIN 论文的候选相关兴趣聚合](https://arxiv.org/abs/1706.06978)，使用 PReLU，
未复现 Dice 和 mini-batch aware regularization，不声称完整复现论文实验。

历史严格满足 `timestamp < cutoff`，先过滤完播，再保留每用户最近 50 个事件；
完全重复事件去重，不同时间的重复观看保留。候选和历史使用相同词表，原始视频 ID 0 是合法视频，
内部 0 为 padding、1 为未知视频；空历史聚合为零。新用户通过热门召回兜底，再走粗排与 DIN。
数值特征执行 signed-log 变换、标准化并追加缺失标记，训练集拟合的词表与标准化状态随模型保存。
验证阶段不更新这些状态，最终重训时才纳入验证窗口。

DIN 训练使用粗排章节的三个训练窗口，以及 8 月 27—29 日验证窗口；直接复用历史召回候选中
有首次观看标签的样本。未观看候选仍保持未知，不作为负例。**历史训练样本不经最终 LightGBM
筛选**，因为该模型训练时见过后续窗口标签；这样避免用未来粗排模型构造历史训练集。
这也意味着训练分布是召回候选，而服务分布是粗排 Top-200，存在需要后续验证的分布差异。
没有把粗排分数加入 DIN 特征。

默认最多 10 轮，以验证 Logloss 早停（耐心值 3），随后从相同随机种子重新初始化，
用三个训练窗口加验证窗口按最佳轮数重训。测试仅由 `--stage test` 或 `--stage all` 显式触发，
不参与早停、词表拟合或标准化。推荐时历史固定为 8 月 29 日截点，不读取测试交互。

### 运行

需要先完成 LightGBM 的 `--stage all`，以建立历史标签特征和最终候选缓存。
当前本机这些文件已存在，继续使用 `D:/Anaconda/envs/recommend/python.exe` 的 CUDA PyTorch。
默认 CUDA，不可用时报错；CPU 需要显式 `--device cpu`。batch size 默认 512，显存不足可减小。

```powershell
# 只训练并保存，不读取 test.csv。
& 'D:/Anaconda/envs/recommend/python.exe' -X utf8 -u -m ranking.din_pipeline --stage train

# 完成训练并在固定测试集评估一次，同候选池比较 DIN 与 LightGBM 的 Top-20。
& 'D:/Anaconda/envs/recommend/python.exe' -X utf8 -u -m ranking.din_pipeline --stage all

# 直接加载已训练模型：三路召回 → LightGBM Top-200 → DIN Top-20。
& 'D:/Anaconda/envs/recommend/python.exe' -X utf8 -m ranking.din_pipeline --stage recommend --user-id 0 --top-k 20

# 冷启动用户走热门候选兜底。
& 'D:/Anaconda/envs/recommend/python.exe' -X utf8 -m ranking.din_pipeline --stage recommend --user-id 999999 --top-k 20

# 单元测试。
& 'D:/Anaconda/envs/recommend/python.exe' -X utf8 -m unittest discover -s tests -v
```

输出目录为 `artifacts/ranking/din/`，包含 `din.pt`、`model_meta.json`、`selection.json`、
`validation_metrics.json`、`test_metrics_k20.json`、历史样本缓存及用户推荐 CSV。
修改训练超参数应使用新的 `--output`；模型拒绝覆盖已有 checkpoint，配置与源数据不匹配时拒绝复用缓存。
checkpoint 保存推理权重、配置、词表和预处理参数，不包含优化器状态；中断后可复用样本缓存从头训练。
CLI 推荐还需要原来的数据和粗排缓存，checkpoint 本身不打包这些文件。

Python 接口：`DINFineRanker.load(path, device='cuda')` 加载；
`predict(candidates, features, history_users, history)` 批量评分；
`recommend(user_id, candidates, feature_store, history_users, history, top_k=20, coarse_ranker=coarse)`
先筛选粗排候选再精排。不传 `coarse_ranker` 时，调用者应传入已经粗排的候选。

测试排名指标沿用窗口内任意一次完播的用户视频正例，分母包含已看和冷启动正例。
DIN 和 LightGBM 的 Top-20 都从同一个粗排 Top-200 池中产生；精排不能恢复粗排已丢弃的正例。
当 `top-k` 与 `coarse-k` 都为 200 时，两者的 Recall 必然相同，只有顺序和 NDCG 可能不同。

### 首次 DIN GPU 训练结果（2026-10-08）

已在 RTX 3050 Ti 上完成真实数据训练。三个训练窗口共 1,300,083 条有标签样本，
验证窗口 116,811 条。第 1 轮验证 Logloss 最优，第 4 轮触发早停；
最佳验证 AUC **0.776506**、Logloss **0.524442**。
随后从相同种子重新初始化，在全部 **1,416,894** 条样本上固定重训 1 轮，
保存 `artifacts/ranking/din/din.pt`（2,127,453 字节）。

固定测试集包含 6,930 位有正例用户、707,227 个用户视频正例；两种方法在同一个
LightGBM Top-200 池中取 Top-20：

| 方法 | 测试用户平均 Recall@20 | 测试 NDCG@20 | 命中正例 |
| --- | ---: | ---: | ---: |
| LightGBM 原顺序 | 0.9471% | **0.050572** | 6,677 |
| DIN 精排 | **1.0476%** | 0.045332 | **7,149** |

DIN 命中数量和 Recall 提高，但 NDCG 下降，**不能据此宣称整体排序质量提升**。
当前保留为已跑通的 DIN 基线，未根据测试结果重新挑选超参数。
粗排候选池包含 87,518 个测试正例，池的用户平均 Recall 上限为 11.5139%；
这里的 Top-20 结果不能直接与前面 Top-200 结果比较。
完整指标见 `artifacts/ranking/din/test_metrics_k20.json`，训练日志见同目录 `train.log`。
14 项单元测试通过。重新加载 checkpoint 后，用户 `0` 与新用户 `999999` 均返回 20 个
不重复、无已看且属于粗排候选池的视频；用户 `0` 的 CPU/GPU 分数最大差为约 `1.19e-7`。
检查记录保存在同目录 `verification.json`，推荐结果保存在 `recommendations_user_0.csv`
和 `recommendations_user_999999.csv`。

## RankMixer 精排对照

新增 `ranking/rankmixer_ranker.py`，在同一条链路中用 **RankMixer 替换 DIN 精排**：
三路召回合并 → LightGBM Top-200 → RankMixer Top-20。原 DIN checkpoint 保留作对照。

实现依据 [RankMixer 论文 §3.1–3.3](https://arxiv.org/html/2507.15551v3)，
采用适合本机 4GB GPU 的小型 Dense 版本，不包含论文的 Sparse-MoE、分布式训练或工业级参数规模。
历史兴趣部分继续使用相同 DIN 局部激活编码器；本次替换的是后续的特征交互网络，
不是删掉历史兴趣信息，也不声称完全复现论文的工业模型。

- 将用户 ID、候选视频 ID、候选相关历史兴趣分别投影为 3 个 token。
- 把现有数值特征分为召回信号、用户统计、视频统计、视频类别、交叉特征 5 组；
  每组的数值与缺失标记一起投影，88 个字段无遗漏、无重复分配。
- 总计 8 个 token，每个 64 维；多头混合将每个 token 切为 8 个 head，交换 token/head 轴后重组，
  该操作无可训练参数，不是自注意力或普通共享 MLP。
- 每个 token 使用独立的两层 FFN（扩展倍数 2，GELU），配残差、post-LayerNorm 和 dropout；
  默认 2 个 block，最终 mean pooling 后输出完播 logit。

默认学习率 0.001、AdamW、batch size 512、最大历史长度 50、最多 10 轮、耐心值 3、随机种子 42，
与 DIN 的训练和标签设置一致。共用 `ranking/neural_ranker.py` 的评分、保存加载和训练步骤，
`ranking/din_pipeline.py` 作为共用实验流程；原 DIN Python 接口与旧 checkpoint 继续兼容。

训练复用相同历史快照；若旧 DIN 样本缓存存在且来源与配置一致，只读复用，否则在 RankMixer
自己的输出目录重建。验证集选择最佳 Logloss 的轮数，然后从相同种子重新初始化，
将训练与验证窗口合并重训。测试时在**同一粗排候选池**里同时评分 RankMixer、原 DIN 和 LightGBM，
输出同口径 Recall@20、NDCG@20、命中数和批量评分耗时，不使用测试结果调参。

```powershell
# 训练、验证选轮数、重训，最后同时评估三个排序方法。
& 'D:/Anaconda/envs/recommend/python.exe' -X utf8 -u -m ranking.rankmixer_pipeline --stage all

# 使用 RankMixer 给用户 0 推荐。
& 'D:/Anaconda/envs/recommend/python.exe' -X utf8 -m ranking.rankmixer_pipeline --stage recommend --user-id 0 --top-k 20

# 原入口也支持显式切换；省略 --model 仍为 DIN。
& 'D:/Anaconda/envs/recommend/python.exe' -X utf8 -m ranking.din_pipeline --model rankmixer --stage recommend --user-id 0
```

新模型默认保存在 `artifacts/ranking/rankmixer/rankmixer.pt`，配套 `selection.json`、
`validation_metrics.json`、`test_metrics_k20.json` 与 `train.log`。
`--stage train` 不读取测试集；`--stage test` 加载已训练模型。RankMixer 测试默认加载
`artifacts/ranking/din/din.pt` 作为基线，可用 `--din-checkpoint` 指定。
调整 `--token-dim`、`--layers`、`--expansion` 或训练参数时，需要新的 `--output` 目录。
两种模型预测分数只用于各自排序，不相加；推荐仍通过热门候选处理新用户。

### 首次 RankMixer GPU 对照结果（2026-10-12）

同样使用 1,300,083 条训练样本、116,811 条验证样本。第 1 轮验证 Logloss 最好，
第 4 轮触发早停；最佳验证 AUC **0.778402**、Logloss **0.523665**，
对应 DIN 为 0.776506、0.524442。随后在全部 1,416,894 条样本上重新训练 1 轮。
最终 RankMixer 共 **759,138** 个参数，checkpoint 大小 3,101,023 字节。

以下是同次评估、同一粗排 Top-200 池、同一个固定测试集的结果；
包括 6,930 位有正例用户、707,227 个用户视频正例：

| 精排方式 | 用户平均 Recall@20 | NDCG@20 | 命中正例 |
| --- | ---: | ---: | ---: |
| LightGBM 原顺序 | 0.9471% | 0.050572 | 6,677 |
| DIN | 1.0476% | 0.045332 | 7,149 |
| RankMixer | **1.1798%** | **0.050662** | **7,604** |

RankMixer 相对 DIN 的 Recall@20 提升 **12.62%**，NDCG@20 提升 **11.76%**。
与 LightGBM 相比，NDCG 的绝对增量只有约 0.000090，不能据此声称显著提升。
本次是固定种子、单次模型配置的离线探索结果，没有进行显著性检验，未根据测试结果再次调参。
本轮 DIN 和 LightGBM 的全部排名指标与原记录完全一致，确认共用流程改动没有改变旧模型结果。

本机离线批量评分约为 RankMixer 21.05 秒、DIN 16.80 秒，包含输入编码与设备传输，
不代表线上单请求延迟，也不是严格隔离其他进程的性能基准。
完整对照见 `artifacts/ranking/rankmixer/test_metrics_k20.json`。
21 项单元测试通过；真实用户 `0` 与冷启动用户 `999999` 均输出 20 个不重复、无已看且属于
粗排候选池的视频，CPU/GPU 分数最大差约 `1.19e-7`。结果位于同目录
`verification.json`、`recommendations_user_0.csv` 和 `recommendations_user_999999.csv`。

## 推荐链路耗时实验

`scripts/benchmark_recommendation.py` 使用现有 checkpoint 实测各阶段耗时，不重新训练模型，
不读取测试标签。运行前需要准备上述模型和候选缓存：

```powershell
& 'D:/Anaconda/envs/recommend/python.exe' -X utf8 -u scripts/benchmark_recommendation.py --users 32 --repeats 3 --warmups 3
```

默认在 `artifacts/benchmarks/recommendation_日期_时间/` 保存 `summary.json` 与逐请求的
`requests.csv`。实验区分首次初始化、模型与数据常驻后的请求、新用户兜底；CUDA 在计时边界同步。
缓存路径沿用当前推荐入口；实时路径串行运行三路各 Top-200，再走相同粗排与精排接口。
分阶段推荐结果会与原 `recommend()` 接口核对。

2026-11-14 在本机 RTX 3050 Ti 上实测，PyTorch 使用 CUDA，CPU 线程数为 4，
LightGBM 预测在 CPU 执行。32 个普通用户各重复 3 次，每种链路共 96 个串行请求：

| 链路 | 中位耗时 | P95 |
| --- | ---: | ---: |
| 候选缓存 → LightGBM → DIN | 46.71 ms | 52.76 ms |
| 候选缓存 → LightGBM → RankMixer | 49.23 ms | 53.66 ms |
| 三路实时召回 → LightGBM → DIN | 55.30 ms | 61.49 ms |
| 三路实时召回 → LightGBM → RankMixer | 58.64 ms | 63.99 ms |

实时 RankMixer 链路的阶段中位耗时：Hot 0.45 ms、ItemCF 1.19 ms、gSASRec 6.61 ms，
候选合并 2.79 ms、特征转换 9.83 ms、LightGBM 评分 0.94 ms、精排接口 35.48 ms。
精排接口包含约 25.04 ms 的编码，以及约 10.10 ms 的网络计算与传输；当前实现每次请求
会重新编码全体用户历史。分阶段中位数不能直接相加作为请求中位数。

实验进程从导入模块到首次缓存 RankMixer 请求完成约 41.07 秒，其中特征快照构建
23.44 秒、用户历史构建 12.45 秒。当前 CLI 每次启动都会执行这些准备；普通用户读取
离线候选缓存，不实时调用三路召回。首次耗时是一次观测，未控制操作系统磁盘缓存；
上述常驻请求时间不包含进程启动、服务网络或并发请求成本。

## 来源

- [官方仓库和字段说明](https://github.com/chongminggao/KuaiRec)
- [固定的数据发布版本](https://zenodo.org/records/18164998)
- [论文：KuaiRec: A Fully-Observed Dataset and Insights for Evaluating Recommender Systems](https://doi.org/10.1145/3511808.3557220)

项目代码许可证见 `LICENSE`；数据的许可和引用要求以官方数据发布页及仓库说明为准。
