# Security Notes

Serve Optimize has a small default Python dependency surface. The heavy GPU serving stacks are optional install profiles and should be audited separately before production use.

## Audit commands

Run the project environment audit:

    uvx pip-audit --progress-spinner off --strict

Run the pinned backend profile audits:

    uvx pip-audit --progress-spinner off --strict --disable-pip --no-deps -r requirements/constraints/vllm.txt
    uvx pip-audit --progress-spinner off --strict --disable-pip --no-deps -r requirements/constraints/sglang.txt

The backend profile commands audit the listed pins directly. A full resolver audit may require the host python3.12 venv package because pip audit creates a temporary virtual environment.

## Current advisory boundary

Last checked: 2026-08-14.

The default project environment reported no known vulnerabilities.

The vLLM 0.27.1 profile reported no known vulnerabilities. The SGLang 0.5.17 profile audit reported advisories in three upstream dependencies:

| Profile | Package | Version | Advisory status |
| --- | --- | --- | --- |
| SGLang | diskcache | 5.6.3 | `PYSEC-2026-2447`; no fixed version was published by the audit source. |
| SGLang | setuptools | 81.0.0 | `PYSEC-2026-3447`; the fixed release conflicts with Torch 2.11.0's required Setuptools range. |
| SGLang | torch | 2.11.0 | `PYSEC-2025-194`; pip audit lists `2.13.0` as the fix version |

The supported vLLM 0.27.1 profile uses Torch 2.13.0. SGLang 0.5.17 still requires Torch 2.11.0 and Setuptools below 82, while its diskcache dependency has no published fix. These advisories remain explicit upstream dependency limits rather than silently relaxed pins.

Treat backend profiles as production dependencies owned jointly with their upstream projects. Reaudit before deployment, and update the profile pins as soon as upstream publishes compatible fixed releases.
