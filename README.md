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

## 来源

- [官方仓库和字段说明](https://github.com/chongminggao/KuaiRec)
- [固定的数据发布版本](https://zenodo.org/records/18164998)
- [论文：KuaiRec: A Fully-Observed Dataset and Insights for Evaluating Recommender Systems](https://doi.org/10.1145/3511808.3557220)

项目代码许可证见 `LICENSE`；数据的许可和引用要求以官方数据发布页及仓库说明为准。
