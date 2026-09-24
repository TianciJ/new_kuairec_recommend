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

## 来源

- [官方仓库和字段说明](https://github.com/chongminggao/KuaiRec)
- [固定的数据发布版本](https://zenodo.org/records/18164998)
- [论文：KuaiRec: A Fully-Observed Dataset and Insights for Evaluating Recommender Systems](https://doi.org/10.1145/3511808.3557220)

项目代码许可证见 `LICENSE`；数据的许可和引用要求以官方数据发布页及仓库说明为准。
