"""
准备 pilot 数据集索引

从 huggingface 下载 RealWorldQA 和 MathVista (testmini)，
图像保存到本地，索引写到 data/pilot/pilot_index.jsonl。

跑法：
    cd /data/zhoukeru/q/TempoMoE
    python scripts/01_prepare_pilot.py

可选参数：
    --rwqa_limit, --mvista_limit 覆盖 yaml 配置
    --only realworldqa | mathvista
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "pilot.yaml"))
    p.add_argument("--rwqa_limit", type=int, default=None)
    p.add_argument("--mvista_limit", type=int, default=None)
    p.add_argument("--only", choices=["realworldqa", "mathvista", None], default=None)
    return p.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    rwqa_n = args.rwqa_limit or cfg["sampling"]["realworldqa_limit"]
    mvista_n = args.mvista_limit or cfg["sampling"]["mathvista_limit"]
    seed = cfg["sampling"]["random_seed"]
    data_root = cfg["paths"]["data_root"]
    out_path = cfg["paths"]["pilot_index"]

    from src.data_loader import (
        prepare_realworldqa, prepare_mathvista, write_pilot_index
    )

    records = []
    if args.only in (None, "realworldqa"):
        records.extend(prepare_realworldqa(data_root, rwqa_n, seed))
    if args.only in (None, "mathvista"):
        records.extend(prepare_mathvista(data_root, mvista_n, seed))

    write_pilot_index(records, out_path)
    print(f"[OK] pilot_index 写入完成: {out_path}")
    print(f"     总条数: {len(records)}")


if __name__ == "__main__":
    main()
