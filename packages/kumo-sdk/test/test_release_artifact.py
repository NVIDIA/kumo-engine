# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import zipfile
from pathlib import Path

import pytest

from scripts.release_artifact import validate_wheel

ROOT = Path(__file__).parents[1]
REPO_ROOT = Path(__file__).parents[3]


def _wheel(tmp_path: Path, *, version: str = '3.0.0b1',
           metadata_version: str | None = None,
           api_requirement: str = 'kumo-api==3.0.0') -> Path:
    path = tmp_path / f'kumoai_studio-{version}-py3-none-any.whl'
    metadata = f'''Metadata-Version: 2.4
Name: kumoai-studio
Version: {metadata_version or version}
Requires-Dist: {api_requirement}
'''
    with zipfile.ZipFile(path, 'w') as wheel:
        wheel.writestr(f'kumoai_studio-{version}.dist-info/METADATA', metadata)
    return path


def test_validates_release_wheel(tmp_path):
    validate_wheel(_wheel(tmp_path), 'kumo-sdk-v3.0.0b1')


@pytest.mark.parametrize(('version', 'metadata_version', 'api_requirement'), [
    ('3.0.0', None, 'kumo-api==3.0.0'),
    ('3.0.0b1', '3.0.0', 'kumo-api==3.0.0'),
    ('3.0.0b1', None,
     'kumo-api @ git+https://github.com/kumo-ai/kumo-api.git@deadbeef'),
    ('3.0.0b1', None, 'kumo-api<4.0.0,>=3.0.0'),
])
def test_rejects_invalid_release_contract(tmp_path, version, metadata_version,
                                          api_requirement):
    with pytest.raises(ValueError):
        validate_wheel(
            _wheel(tmp_path, version=version,
                   metadata_version=metadata_version,
                   api_requirement=api_requirement), 'kumo-sdk-v3.0.0b1')


def test_release_retains_one_validated_artifact():
    build = (REPO_ROOT /
             '.github/workflows/kumo-sdk-build-wheels.yml').read_text()
    release = (REPO_ROOT /
               '.github/workflows/kumo-sdk-release-on-tag.yml').read_text()
    upload = (REPO_ROOT /
              '.github/workflows/kumo-sdk-upload-wheels.yml').read_text()
    validator = (ROOT / 'scripts/validate_wheels_for_release.sh').read_text()
    artifact_name = 'wheels-${{ github.sha }}-${{ github.run_id }}'

    assert release.count(f'artifact-name: {artifact_name}') >= 2
    assert 'retention-days: 14' in release
    assert 'expected-version: ${{ github.ref_name }}' in release
    assert 'name: ${{ inputs.artifact-name }}' in upload
    assert 'path: ./packages/kumo-sdk/release-artifact' in upload
    assert 'path: ./packages/kumo-sdk/dist/*.whl' in build
    assert 'Expected exactly one wheel' in validator


def test_release_dependency_pins_api_beta():
    setup = (ROOT / 'setup.py').read_text()

    assert "kumo_api_release_ver = '3.0.0'" in setup
    assert "kumo_api = f'kumo-api=={api_ver}'" in setup
    assert 'git+https://github.com/kumo-ai/kumo-api' not in setup


def test_public_sdk_does_not_inherit_the_application_fsspec_cap():
    setup = (ROOT / 'setup.py').read_text()

    assert '"fsspec>=2025.3.0"' in setup
    assert 'fsspec>=2025.3.0,<=' not in setup


def test_publication_keeps_s3_out_of_the_public_path():
    release = (REPO_ROOT /
               '.github/workflows/kumo-sdk-release-on-tag.yml').read_text()
    upload = (REPO_ROOT /
              '.github/workflows/kumo-sdk-upload-wheels.yml').read_text()
    nightly = (REPO_ROOT /
               '.github/workflows/kumo-sdk-release-nightly.yml').read_text()
    testpypi_job = release.split('  stage-testpypi:',
                                 1)[1].split('  publish-pypi:', 1)[0]

    testpypi = upload.index('Upload wheels to TestPyPI')
    pypi = upload.index('Upload wheels to PyPI')
    s3 = upload.index('Upload wheels to internal S3')
    assert testpypi < pypi < s3
    assert release.index('stage-testpypi:') < release.index('publish-pypi:')
    assert release.index('publish-pypi:') < release.index('create-release:')
    assert release.count('allow-s3-failure: true') == 1
    assert upload.count(
        'continue-on-error: ${{ inputs.allow-s3-failure }}') == 2
    assert upload.count('skip-existing: true') == 2
    assert 'id-token: write' in testpypi_job
    assert 'allow-s3-failure: true' not in nightly
    assert 'upload-to-pypi: false' in nightly
    assert "vars.KUMO_SDK_PUBLICATION_ENABLED == 'true'" in upload
