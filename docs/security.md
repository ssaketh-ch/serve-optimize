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

Last checked: 2026-08-09.

The default project environment reported no known vulnerabilities.

The optional backend profile audits reported upstream advisories in the currently available serving stacks:

| Profile | Package | Version | Advisory status |
| --- | --- | --- | --- |
| vLLM | torch | 2.11.0 | `PYSEC-2025-194`; pip audit lists `2.13.0` as the fix version |
| SGLang | torch | 2.11.0 | `PYSEC-2025-194`; pip audit lists `2.13.0` as the fix version |

The supported vLLM 0.24.0 and SGLang 0.5.13.post1 profiles currently pin torch 2.11.0. Moving either profile to torch 2.13.0 requires dependency resolution, backend capability checks, and live validation before it can replace the supported pin.

Treat backend profiles as production dependencies owned jointly with their upstream projects. Reaudit before deployment, and update the profile pins as soon as upstream publishes compatible fixed releases.
