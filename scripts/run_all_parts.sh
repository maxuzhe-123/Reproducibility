#!/usr/bin/env bash
set -euo pipefail
cd "xxxx/xxxx"

python -m src.pipeline_v2.preds --config configs/codejudgebench_part_01.yaml > scripts/codejudgebench_part_01.log 2>&1 &
python -m src.pipeline_v2.preds --config configs/codejudgebench_part_02.yaml > scripts/codejudgebench_part_02.log 2>&1 &
python -m src.pipeline_v2.preds --config configs/codejudgebench_part_03.yaml > scripts/codejudgebench_part_03.log 2>&1 &
python -m src.pipeline_v2.preds --config configs/codejudgebench_part_04.yaml > scripts/codejudgebench_part_04.log 2>&1 &
python -m src.pipeline_v2.preds --config configs/codejudgebench_part_05.yaml > scripts/codejudgebench_part_05.log 2>&1 &
python -m src.pipeline_v2.preds --config configs/codejudgebench_part_06.yaml > scripts/codejudgebench_part_06.log 2>&1 &
python -m src.pipeline_v2.preds --config configs/codejudgebench_part_07.yaml > scripts/codejudgebench_part_07.log 2>&1 &
python -m src.pipeline_v2.preds --config configs/codejudgebench_part_08.yaml > scripts/codejudgebench_part_08.log 2>&1 &
python -m src.pipeline_v2.preds --config configs/codejudgebench_part_09.yaml > scripts/codejudgebench_part_09.log 2>&1 &
python -m src.pipeline_v2.preds --config configs/codejudgebench_part_10.yaml > scripts/codejudgebench_part_10.log 2>&1 &
python -m src.pipeline_v2.preds --config configs/codejudgebench_part_11.yaml > scripts/codejudgebench_part_11.log 2>&1 &
python -m src.pipeline_v2.preds --config configs/codejudgebench_part_12.yaml > scripts/codejudgebench_part_12.log 2>&1 &
python -m src.pipeline_v2.preds --config configs/codejudgebench_part_13.yaml > scripts/codejudgebench_part_13.log 2>&1 &
python -m src.pipeline_v2.preds --config configs/codejudgebench_part_14.yaml > scripts/codejudgebench_part_14.log 2>&1 &
python -m src.pipeline_v2.preds --config configs/codejudgebench_part_15.yaml > scripts/codejudgebench_part_15.log 2>&1 &

wait
echo "all parts done"
