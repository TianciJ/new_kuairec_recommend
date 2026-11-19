"""RankMixer 入口，复用 DIN 的数据、训练与同候选池对照评估。"""
from ranking.din_pipeline import main


if __name__ == '__main__':
    main(default_model='rankmixer')
