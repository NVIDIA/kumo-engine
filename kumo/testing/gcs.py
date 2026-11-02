# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import inspect
import json
import os
from collections.abc import Callable
from unittest import mock

import pytest
from google.cloud import storage

from io_utils import logging
from io_utils.fsutil import impl_gcs
from kumo.connector.bigquery_connector import (
    ENV_GOOGLE_CREDENTIALS_DEFAULT_PATH,
    ENV_GOOGLE_CREDENTIALS_JSON,
)

logger = logging.getLogger(__name__)

# Environment variable for GCS test bucket
ENV_GCS_TEST_BUCKET = "GCS_TEST_BUCKET"

# Default Application Default Credentials path
ADC_DEFAULT_PATH = os.path.join(os.path.expanduser("~"), ".config", "gcloud",
                                "application_default_credentials.json")


def setup_gcs_credentials():
    """Set up GCS credentials for testing.

    This function fetches Google Cloud credentials from the
    GOOGLE_CREDENTIALS_STR_JSON environment variable and sets up the necessary
    environment variables for GCS testing.

    The credentials should be base64 encoded to avoid issues with special
    characters in the JSON string.
    """

    if ENV_GCS_TEST_BUCKET not in os.environ:
        os.environ[ENV_GCS_TEST_BUCKET] = 'kumo-test-bucket'
    google_creds = os.getenv("GOOGLE_CREDENTIALS_STR_JSON")
    if google_creds:
        logger.info("Using GOOGLE_CREDENTIALS_STR_JSON environment variable")
        google_creds = base64.b64decode(google_creds)
        google_creds = google_creds.decode('utf-8')

        file_path = os.path.join(os.getcwd(), ".google_credential.json")
        # Parse the JSON string to a dictionary first
        try:
            google_creds_dict = json.loads(google_creds)
            with open(file_path, 'w') as f:
                json.dump(google_creds_dict, f)

            with open(ENV_GOOGLE_CREDENTIALS_DEFAULT_PATH, 'w') as f:
                json.dump(google_creds_dict, f)

            # Extract and set GCP_PROJECT_ID from service account credentials
            # This is needed for GCPSecretProvider initialization
            if 'project_id' in google_creds_dict:
                os.environ['GCP_PROJECT_ID'] = google_creds_dict['project_id']
                logger.info(f"Set GCP_PROJECT_ID from credentials: "
                            f"{google_creds_dict['project_id']}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse GOOGLE_CREDENTIALS_STR_JSON: {e}")
            raise
        os.environ[ENV_GOOGLE_CREDENTIALS_JSON] = file_path
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = file_path
    else:
        logger.warning(
            "GOOGLE_CREDENTIALS_STR_JSON environment variable not set. "
            "Relying on existing GOOGLE_APPLICATION_CREDENTIALS.")


def _check_gcs_credentials() -> bool:
    """Check if GCS credentials are available.

    Checks for credentials in the following order:
    1. GOOGLE_APPLICATION_CREDENTIALS env var
    2. ~/.google_credentials.json (ENV_GOOGLE_CREDENTIALS_DEFAULT_PATH)
    3. ~/.config/gcloud/application_default_credentials.json (ADC default)

    If ADC exists but GOOGLE_APPLICATION_CREDENTIALS is not set, it will be
    set automatically for the google-cloud libraries to use.

    Returns:
        bool: True if credentials are available, False otherwise
    """
    setup_gcs_credentials()

    # Check if credentials are already set
    if os.getenv('GOOGLE_APPLICATION_CREDENTIALS') is not None:
        return True

    if os.path.exists(ENV_GOOGLE_CREDENTIALS_DEFAULT_PATH):
        return True

    # Check for ADC default path and set env var if found
    if os.path.exists(ADC_DEFAULT_PATH):
        logger.info(f"Using ADC credentials from {ADC_DEFAULT_PATH}")
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = ADC_DEFAULT_PATH
        return True

    return False


def withGCSCredentials(func: Callable):
    """Decorator that sets up GCS credentials and skips tests if unavailable.

    This decorator:
    1. Calls setup_gcs_credentials() to fetch and configure credentials
    2. Skips the test if credentials are not available
    3. Allows tests to run in CI with credentials but skip locally without them

    Args:
        func: The test function to decorate

    Example:
        @withGCSCredentials
        def test_gcs_operation():
            # Test code here
            pass
    """
    # A collection-time mark prevents live GCS fixtures from being resolved
    # when credentials are absent. A runtime wrapper skips too late: pytest
    # sets up fixtures before invoking the wrapped test.
    from kumo.testing._skip import requires_resource
    func = requires_resource('gcp', func)
    credentials_missing = not _check_gcs_credentials()
    reason = "No GCS credentials found"
    if inspect.isgeneratorfunction(func):
        from kumo.testing._skip import skip_if
        return skip_if(credentials_missing, reason, func)
    return pytest.mark.skipif(credentials_missing, reason=reason)(func)


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
