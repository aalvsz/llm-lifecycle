# Security policy

## Supported versions

This project is currently an experimental alpha. Security fixes are applied to the latest `main` revision.

## Reporting a vulnerability

Please use GitHub's private vulnerability reporting for this repository. Do not open a public issue containing
credentials, exploit details, private model or dataset paths, or infrastructure information.

The lifecycle runner never invokes a shell, but backend commands execute third-party training code and may load
remote models or datasets. Review generated manifests and use only trusted checkpoints, repositories, container
images, and remote-code implementations.
