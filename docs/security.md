# Security Notes

Serve Optimize has a small default Python dependency surface. The heavy GPU serving stacks are optional install profiles and should be audited separately before production use.

## Audit commands

Run the project environment audit:

    uvx pip-audit --progress-spinner off --strict

Run the pinned backend profile audits:

    uvx pip-audit --progress-spinner off --strict --disable-pip --no-deps -r requirements/constraints/vllm.txt
    uvx pip-audit --progress-spinner off --strict --disable-pip --no-deps -r requirements/constraints/sglang.txt

The backend profile commands audit the listed pins directly. A full resolver audit may require the host python3.12 venv package because pip audit creates a temporary virtual environment.

Run complete resolved environment audits from Bash, excluding the unpublished local Serve Optimize wheel:

    uvx pip-audit --progress-spinner off --strict --disable-pip --no-deps -r <(uv pip freeze --python .venv-vllm/bin/python | sed '/^serve-optimize/d')
    uvx pip-audit --progress-spinner off --strict --disable-pip --no-deps -r <(uv pip freeze --python .venv-sglang/bin/python | sed '/^serve-optimize/d')

## Current advisory boundary

Last checked: 2026-08-14.

The default project environment reported no known vulnerabilities.

The complete vLLM 0.27.1 and SGLang 0.5.17 environment audits reported these upstream dependency advisories:

| Profile | Package | Version | Advisory status |
| --- | --- | --- | --- |
| vLLM | setuptools | 80.10.2 | `PYSEC-2026-3447`; vLLM requires Setuptools below 81 on Python 3.12, while the fixed release is 83.0.0. |
| SGLang | diskcache | 5.6.3 | `PYSEC-2026-2447`; no fixed version was published by the audit source. |
| SGLang | setuptools | 81.0.0 | `PYSEC-2026-3447`; the fixed release conflicts with Torch 2.11.0's required Setuptools range. |
| SGLang | torch | 2.11.0 | `PYSEC-2025-194`; pip audit lists `2.13.0` as the fix version |

The supported vLLM 0.27.1 profile uses Torch 2.13.0 but requires Setuptools below 81 on Python 3.12. SGLang 0.5.17 still requires Torch 2.11.0 and Setuptools below 82, while its diskcache dependency has no published fix. These advisories remain explicit upstream dependency limits rather than silently relaxed pins.

Treat backend profiles as production dependencies owned jointly with their upstream projects. Reaudit before deployment, and update the profile pins as soon as upstream publishes compatible fixed releases.
