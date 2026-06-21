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
