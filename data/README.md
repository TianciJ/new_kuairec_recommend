# 数据目录

- `sources/kuairec.json`：固定版本的官方下载地址、官方文件 MD5，以及从已校验压缩包解出的文件的本地 MD5。
- `raw/KuaiRec 2.0/data/`：当前保留 5 份交互、视频、文本和社交数据。
- `raw/KuaiRec 2.0/LICENSE`：数据许可证。
- `raw/supplemental/`：用户原始特征和视频原始分类，共 2 份 CSV。
- `processed/`：固定时间划分后的 `train.csv`、`test.csv`、划分元数据 `split_metadata.json` 和视频范围过滤记录 `video_id_filter_metadata.json`。

数据文件不提交到 Git。如需重新获取，可通过 `sources/kuairec.json` 中的官方地址下载；该文件中的校验值始终指向官方下载版本。

## big_matrix 本地清理记录

后续推荐实验只使用 `big_matrix.csv` 作为交互数据源。按用户要求，已在该文件中删除 `date` 与 `timestamp` 转为北京时间（UTC+8）后的日期不一致的记录。

- 清理前：12,530,806 条；删除：15,530 条；清理后：12,515,276 条。
- 官方原版 MD5：`080475f93f60916336ad6b80f102d2cc`。
- 日期清理后 MD5（历史版本）：`7b5213afb61fd3ffbe19ae43731758e5`。
- 该步骤保留行的内容和顺序不变；后续视频范围过滤结果见下文。当前目录未保留原始备份。

## 固定训练与测试划分

已使用清理后的 `big_matrix.csv`，以北京时间 `2020-08-29 00:00:00+08:00`（Unix 时间戳 `1598630400`）为分界生成以下文件。整天截断对应约 80/20，不进行随机抽样。

| 文件 | 划分条件 | 交互数 | 占比 |
| --- | --- | ---: | ---: |
| `processed/train.csv` | `timestamp < 1598630400` | 9,923,901 | 79.4630% |
| `processed/test.csv` | `timestamp >= 1598630400` | 2,564,808 | 20.5370% |

两份文件保留原始 8 列及各自记录的原有顺序。最初仅按时间划分，之后与 `big_matrix.csv` 同步移除原始视频分类表未覆盖的 45 个视频；当前合计 12,488,709 条，每条源记录恰好分入一份文件。过滤时核验时间边界，并通过分区内容校验值确认 train/test 与过滤后的源文件完全对应。

`processed/split_metadata.json` 保存当前源文件及输出文件 MD5、时间边界、各文件记录数、时间范围和视频过滤条件。

## item_daily_features 列精简

按用户要求直接修改 `raw/KuaiRec 2.0/data/item_daily_features.csv`，不保留原版备份。第一轮删除以下 15 列，字段数从 58 减至 43：

- `video_width`、`video_height`、`upload_type`
- `click_like_cnt`、`double_click_cnt`
- `direct_comment_cnt`、`reply_comment_cnt`
- `delete_comment_cnt`、`delete_comment_user_num`
- `comment_like_cnt`、`comment_like_user_num`
- `download_cnt`、`download_user_num`
- `cancel_collect_cnt`、`cancel_collect_user_num`

第二轮按用户要求删除训练期（`date < 20200829`）零值占比超过 50% 的全部 12 列，字段数从 43 减至 31。零值比例以全部训练期记录为分母，缺失值不计为零。

| 删除字段（次数 / 人数） | 训练期零值占比 |
| --- | ---: |
| `cancel_follow_cnt` / `cancel_follow_user_num` | 98.2851% |
| `report_cnt` / `report_user_num` | 97.9271% |
| `share_cnt` / `share_user_num` | 61.1636% |
| `comment_cnt` / `comment_user_num` | 53.8920% |
| `reduce_similar_cnt` / `reduce_similar_user_num` | 53.5224% |
| `collect_cnt` / `collect_user_num` | 50.2705% |

第三轮按用户要求删除 `video_tag_name`，保留 `video_tag_id`，字段数从 31 减至 30。

列精简完成时为 30 列、343,341 行，已逐行核验剩余字段的值和记录顺序不变。三次操作均只删列，不做时间筛选或按日聚合，也没有保留原版备份。后续按视频范围过滤后，当前为 30 列、342,281 行。
列精简完成时 MD5（历史版本）为 `c6a8fc1f562ccce7cbc57ae580d96fd4`；第二轮精简后的 MD5 为 `521b5e5bde497b593e71eb49f43ac7bb`；第一轮精简后的 MD5 为 `38bc00a7123ad15e8fc957ac68785e48`；`sources/kuairec.json` 中的 MD5 仍为官方下载版本的校验值。当前校验值见视频过滤元数据。
若需要被删除的列，须从官方发布版本重新获取。

## video_raw_categories_multi 完全重复记录去重

按用户要求直接修改 `raw/supplemental/video_raw_categories_multi.csv`，以全部 13 列的值完全相同为重复判定，每组只保留首次出现的记录。

- 去重前 26,826 行，删除 55 条重复记录，去重后 26,771 行。
- 仍覆盖 10,683 个视频；同一视频的不同类目标注全部保留。
- 已核验剩余完全重复记录为 0，保留记录的字段值和相对顺序不变。
- 当前 MD5：`bbb16db5e9a97e10854feb4414f2a652`；官方原版 MD5：`d05eea147135d2cdf7759fba5c0d70d4`。
- 未保留原版备份；如需恢复原始文件，可从官方发布版本重新获取。本次未修改分类概率或其他字段值。

## 统一视频 ID 范围

以 `raw/supplemental/video_raw_categories_multi.csv` 中的 10,683 个不同 `video_id` 为准，直接从其他所有含 `video_id` 的现有 CSV 中移除不在该集合内的 45 个视频。保留原始 ID 和记录顺序，不重新编号、不额外删除用户，也不改变训练与测试的时间分界。

| 文件 | 本次删除记录数 | 当前记录数 |
| --- | ---: | ---: |
| `big_matrix.csv` | 26,567 | 12,488,709 |
| `processed/train.csv` | 15,644 | 9,923,901 |
| `processed/test.csv` | 10,923 | 2,564,808 |
| `item_daily_features.csv` | 1,060 | 342,281 |
| `item_categories.csv` | 45 | 10,683 |
| `kuairec_caption_category.csv` | 45 个视频记录 | 10,683 个视频记录 |

以上行均按表各自粒度计数；big_matrix 和 train/test 是同一批交互的两种组织方式，不能把删除数量相加。train/test 分别覆盖 8,905 / 5,622 个视频，均为基准视频集合的子集。

文本表按完整视频记录过滤，包含跨行文案；已有的文本 CSV 格式异常本次保留，未自动修复。视频原始分类表作为基准保持 26,771 行，用户属性和社交表未改动。

完整删除 ID 列表、各表过滤前后校验值及核验结果保存在 `processed/video_id_filter_metadata.json`。执行脚本为 `scripts/filter_video_ids.py`；本次没有生成原始数据备份，恢复被移除的记录需要重新获取官方文件并重做对应清理步骤。
