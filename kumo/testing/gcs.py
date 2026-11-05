# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from collections.abc import Callable, Generator
from contextlib import contextmanager
from unittest import mock

import pytest
from google.cloud import storage

from io_utils.fsutil import impl_gcs
from kumo.testing.bigquery import setup_bigquery_credentials

ENV_GCS_TEST_BUCKET = 'GCS_TEST_BUCKET'


@contextmanager
def setup_gcs_credentials() -> Generator[None, None, None]:
    """Scope explicit ADC without discovering or writing credential files."""
    with setup_bigquery_credentials(), pytest.MonkeyPatch.context() as environment:
        environment.setenv(ENV_GCS_TEST_BUCKET,
                           os.environ.get(ENV_GCS_TEST_BUCKET) or 'kumo-test-bucket')
        # Cached clients otherwise retain an identity from an earlier scope.
        environment.setattr(impl_gcs, 'threadsafe_gcs',
                            impl_gcs.ThreadLocalGCSClient())
        yield


def withGCSCredentials(func: Callable):
    """Declare credentials without performing IO during pytest collection."""
    return pytest.mark.gcs_credentials(func)


def withMockedGCSFilesystem(func: Callable):
    """Decorator to mock GCS filesystem for testing.

    Note: This uses unittest.mock to create a fake GCS client. For more
    comprehensive testing, consider using:
    - google-cloud-testutils
    - gcs-testbench (Google's official test server)
    - fake-gcs-server
    """
    def wrapper():
        # Save GCP environment variables and original threadsafe_gcs instances
        credentials = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
        project = os.environ.get('GCLOUD_PROJECT')
        old_gcs_impl = impl_gcs.threadsafe_gcs

        try:
            # Create a mock storage client
            mock_client = mock.MagicMock(spec=storage.Client)

            # Setup mock buckets dictionary to simulate GCS storage
            mock_buckets = {}

            def get_or_create_bucket(bucket_name, **_kwargs):
                if bucket_name not in mock_buckets:
                    mock_bucket = mock.MagicMock(spec=storage.Bucket)
                    mock_bucket.name = bucket_name
                    mock_bucket._blobs = {}

                    # Mock blob creation
                    def mock_blob(blob_name):
                        if blob_name not in mock_bucket._blobs:
                            blob = mock.MagicMock(spec=storage.Blob)
                            blob.name = blob_name
                            blob.bucket = mock_bucket
                            blob._data = None
                            blob.size = 0
                            blob.updated = None

                            # Mock blob.exists()
                            blob.exists.return_value = False

                            # Mock blob.upload_from_string()
                            def upload_from_string(data, **kwargs):
                                blob._data = data if isinstance(
                                    data, bytes) else data.encode()
                                blob.size = len(blob._data)
                                blob.exists.return_value = True

                            blob.upload_from_string.side_effect = upload_from_string

                            # Mock blob.upload_from_file()
                            def upload_from_file(file_obj, **kwargs):
                                blob._data = file_obj.read()
                                blob.size = len(blob._data)
                                blob.exists.return_value = True

                            blob.upload_from_file.side_effect = upload_from_file

                            # Mock blob.upload_from_filename()
                            def upload_from_filename(filename, **kwargs):
                                with open(filename, 'rb') as f:
                                    blob._data = f.read()
                                blob.size = len(blob._data)
                                blob.exists.return_value = True

                            blob.upload_from_filename.side_effect = upload_from_filename

                            # Mock blob.download_as_bytes()
                            def download_as_bytes():
                                if not blob.exists.return_value:
                                    from google.api_core import exceptions
                                    raise exceptions.NotFound("Blob not found")
                                return blob._data

                            blob.download_as_bytes.side_effect = download_as_bytes

                            # Mock blob.download_to_filename()
                            def download_to_filename(filename):
                                if not blob.exists.return_value:
                                    from google.api_core import exceptions
                                    raise exceptions.NotFound("Blob not found")
                                with open(filename, 'wb') as f:
                                    f.write(blob._data)

                            blob.download_to_filename.side_effect = download_to_filename

                            # Mock blob.delete()
                            def delete():
                                blob._data = None
                                blob.size = 0
                                blob.exists.return_value = False

                            blob.delete.side_effect = delete

                            # Mock blob.reload()
                            def reload():
                                if not blob.exists.return_value:
                                    from google.api_core import exceptions
                                    raise exceptions.NotFound("Blob not found")

                            blob.reload.side_effect = reload

                            mock_bucket._blobs[blob_name] = blob

                        return mock_bucket._blobs[blob_name]

                    mock_bucket.blob.side_effect = mock_blob

                    # Mock list_blobs
                    def list_blobs(prefix='', delimiter=None, max_results=None):
                        blobs = []
                        prefixes = set()

                        for blob_name, blob in mock_bucket._blobs.items():
                            if blob.exists.return_value and blob_name.startswith(
                                    prefix):
                                if delimiter:
                                    # Find the next delimiter after the prefix
                                    remainder = blob_name[len(prefix):]
                                    if delimiter in remainder:
                                        # This is a subdirectory
                                        subdir = remainder.split(
                                            delimiter)[0] + delimiter
                                        prefixes.add(prefix + subdir)
                                    else:
                                        # This is a file in the current directory
                                        blobs.append(blob)
                                else:
                                    blobs.append(blob)

                        # Create mock iterator
                        mock_iterator = mock.MagicMock()
                        mock_iterator.__iter__ = lambda self: iter(
                            blobs[:max_results] if max_results else blobs)
                        mock_iterator.prefixes = list(prefixes)

                        # Mock pages
                        mock_page = mock.MagicMock()
                        mock_page.__iter__ = lambda self: iter(
                            blobs[:max_results] if max_results else blobs)
                        mock_page.prefixes = list(prefixes)
                        mock_iterator.pages = iter([mock_page])

                        return mock_iterator

                    mock_bucket.list_blobs.side_effect = list_blobs

                    # Mock bucket.exists()
                    mock_bucket.exists.return_value = True

                    # Mock bucket.reload()
                    mock_bucket.reload.return_value = None
                    mock_bucket.location = 'US'

                    # Mock copy_blob
                    def copy_blob(source_blob, dest_bucket, dest_blob_name):
                        dest_blob = dest_bucket.blob(dest_blob_name)
                        dest_blob._data = source_blob._data
                        dest_blob.size = source_blob.size
                        dest_blob.exists.return_value = True
                        return dest_blob

                    mock_bucket.copy_blob.side_effect = copy_blob

                    mock_buckets[bucket_name] = mock_bucket

                return mock_buckets[bucket_name]

            mock_client.bucket.side_effect = get_or_create_bucket

            def create_bucket(bucket_name, **kwargs):
                return get_or_create_bucket(bucket_name)

            mock_client.create_bucket.side_effect = create_bucket

            # Mock the threadsafe client to return our mock
            mock_threadsafe = mock.MagicMock()
            mock_threadsafe.client.return_value = mock_client

            impl_gcs.threadsafe_gcs = mock_threadsafe

            # Call the test function with the mock client
            func(mock_client)

        finally:
            # Reset the environment variables back
            if credentials:
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials
            elif 'GOOGLE_APPLICATION_CREDENTIALS' in os.environ:
                del os.environ['GOOGLE_APPLICATION_CREDENTIALS']

            if project:
                os.environ['GCLOUD_PROJECT'] = project
            elif 'GCLOUD_PROJECT' in os.environ:
                del os.environ['GCLOUD_PROJECT']

            # Restore original threadsafe_gcs instances in both modules
            impl_gcs.threadsafe_gcs = old_gcs_impl

    return wrapper


def withGCSEmulator(func: Callable):
    """Decorator to use GCS Storage Emulator for testing.

    This requires the GCS Storage Emulator to be running:
    - Install: npm install -g @google-cloud/storage-emulator
    - Start: gcloud beta emulators storage start

    Or use Docker:
    - docker run -p 9023:9023 fsouza/fake-gcs-server
    """
    def wrapper():
        # Save original environment
        original_host = os.environ.get('STORAGE_EMULATOR_HOST')
        old_gcs_impl = impl_gcs.threadsafe_gcs

        try:
            # Set emulator host
            os.environ['STORAGE_EMULATOR_HOST'] = 'http://localhost:9023'

            # Reset client to pick up emulator settings
            # Create new ThreadLocalGCSClient instance from impl_gcs
            new_client = impl_gcs.ThreadLocalGCSClient()
            impl_gcs.threadsafe_gcs = new_client

            # Create client that connects to emulator
            client = storage.Client(
                project='test-project',
                client_options={'api_endpoint': 'http://localhost:9023'})

            func(client)

        finally:
            # Reset environment
            if original_host:
                os.environ['STORAGE_EMULATOR_HOST'] = original_host
            elif 'STORAGE_EMULATOR_HOST' in os.environ:
                del os.environ['STORAGE_EMULATOR_HOST']

            # Restore original threadsafe_gcs instances in both modules
            impl_gcs.threadsafe_gcs = old_gcs_impl

    return wrapper
