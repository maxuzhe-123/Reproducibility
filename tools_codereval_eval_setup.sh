#!/usr/bin/env bash
# Rebuild the codereval_eval evaluation environment (shared by GT and the three CoderEval experiments).
# Usage: bash tools_codereval_eval_setup.sh
set -e
CONDA=xxxx/xxxx/bin/conda
$CONDA create -n codereval_eval python=3.11 -y
PIP=xxxx/xxxx/envs/codereval_eval/bin/pip
# Evaluation harness dependencies
$PIP install numpy dill
# Third-party dependencies of the CoderEval repositories (converged iteratively from gt_env_error in GroundTruth.py)
$PIP install "setuptools<81" six lxml fs pbr requests identify gevent pyOpenSSL \
  "ruamel.yaml" marshmallow jsonschema pycryptodome bleach PyYAML flexmock \
  python-dateutil pytz isodate argcomplete colorlog Pillow yarl xrd webargs \
  "requests-cache" python-magic python-slugify Jinja2 GitPython commonmark ansible \
  PyGithub fs-s3fs redis aiohttp marshmallow-union dirty-validators pairtree \
  pytest PyLD libmozdata tenacity marshmallow-enum humanize kubernetes_asyncio \
  terminaltables filelock tosca-parser SQLAlchemy picles.plumber lark Faker factory_boy
# Note: use picles.plumber (packtools needs plumber.filter); do not install BlueDynamics' plumber
echo "done. 验证: xxxx/xxxx/envs/codereval_eval/bin/python GroundTruth.py  应得 passed=230 env_error=0"
