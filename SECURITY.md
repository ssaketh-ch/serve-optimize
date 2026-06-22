# Security Policy

## Reporting A Vulnerability

Report suspected vulnerabilities privately through GitHub Security Advisories:

`https://github.com/ssaketh-ch/serve-optimize/security/advisories/new`

Do not open a public issue for a vulnerability and do not include access tokens, model credentials, or private benchmark artifacts in reports.

Include the affected command or component, reproduction steps, impact, and any suggested mitigation. Maintainers will acknowledge a report as soon as practical and coordinate disclosure after a fix is available.

## Supported Versions

Security fixes target the latest release and the current `main` branch.

## Scope Notes

Serve Optimize can launch local backend processes and write benchmark artifacts. Treat model paths, launch arguments, endpoint URLs, evidence databases, and output directories as trusted operator inputs. Review generated campaign command files before running them.

Endpoint requests accept only HTTP and HTTPS base URLs. For authenticated endpoints, pass the environment variable name with `--api-key-env`; the secret value is read when requests are sent and is not written to run artifacts. Pin remote model inputs with `--model-revision` when reproducible supply chain identity is required.
