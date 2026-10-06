# AWS RDS CA bundle

This compressed bundle is copied into the Kumo backend images so AWS RDS TLS
works without installer-managed ConfigMaps or network access during image
builds. It was refreshed from the provider sources on 2026-07-27. Azure
PostgreSQL uses the image's standard OS trust store.

`aws-rds-roots.pem.gz` combines:

- AWS commercial `https://truststore.pki.rds.amazonaws.com/global/global-bundle.pem`
- AWS GovCloud `https://truststore.pki.us-gov-west-1.rds.amazonaws.com/global/global-bundle.pem`

Its compressed SHA-256, enforced by the installer, is
`ff835c838e93979c7f5e4afdc18cbf00c49b685aadb24260747299ba049b7e6a`;
its expanded PEM SHA-256 is
`ade2a007462ac5f0f781d4a9cbfe48a098bb4c50071ce688f00d3f791179d557`.

The installer verifies the compressed SHA-256 value and required certificate
families while building the image. Refresh the file during normal dependency
maintenance when AWS changes its documented trust chain, and update the
compressed digest in this file and `install_postgres_ca_bundles` together.
