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

Last checked: 2026-06-23.

The default project environment reported no known vulnerabilities.

The optional backend profile audits reported upstream advisories in the currently available serving stacks:

| Profile | Package | Version | Advisory status |
| --- | --- | --- | --- |
| vLLM | vllm | 0.23.0 | Reported by pip audit, with no fixed version listed |
| vLLM | torch | 2.11.0 | Reported by pip audit, with no compatible backend supported bump |
| SGLang | torch | 2.11.0 | Reported by pip audit, with no compatible backend supported bump |

The resolver shows no newer vLLM release than 0.23.0 and no newer SGLang release than 0.5.13.post1. Both current backend packages hard pin torch 2.11.0, so forcing torch 2.12.1 would make the supported profiles unsatisfiable.

Treat backend profiles as production dependencies owned jointly with their upstream projects. Reaudit before deployment, and update the profile pins as soon as upstream publishes compatible fixed releases.
