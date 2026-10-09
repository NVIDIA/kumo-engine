# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from http import HTTPStatus
from typing import Any, Dict

import requests
from fastapi import APIRouter, Depends
from kumo_core.version import get_kumo_version

from async_utils.sync import blocking_func
from io_utils import logging
from io_utils.secrets import get_secrets
from rest.config.constants import DebugInfo, FeatureFlags
from rest.deps.authentication import Authentication
from rest.utils.routers import enable_api_caching

log = logging.getLogger(__name__)


class ConstantsResource:
    def __init__(self, auth: Authentication) -> None:
        self.router = APIRouter(
            prefix="/v0/constants",
            tags=["Constants"],
            dependencies=[Depends(auth.authenticate_request_header)],
            responses={
                HTTPStatus.NOT_FOUND: {
                    "description": "Not found (Constants)"
                }
            },
        )

        self.router.add_api_route(
            "/flags",
            self.flags,
            response_model=DebugInfo,
            methods=['GET'],
        )
        # Enable caching for ALL GET apis.
        enable_api_caching(self.router, method='GET')

    def flags(self) -> DebugInfo:
        r"""Return a struct with flags needed to render the UI"""

        # Return data
        return DebugInfo(
            db_path='',
            release_version=get_kumo_version(),
            temporal_url='',
            feature_flags=FeatureFlags.parse_from_env_vars(),
        )


@blocking_func()
def query_readme_docs_api(slug: str) -> Dict[Any, Any]:
    r"""
    Query and return docs subpages from ReadMe docs.

    Args:
        slug (str): Path in external documenation
        matching `https://kumo.ai/docs/reference/{slug}`
        (e.g. https://kumo.ai/docs/reference/assuming).

    Returns:
        Store and return the text response of
        the queried ReadMe page
        as a python dictionary.
        The keys in the dictionary are:
        metadata, api, next, algolia,
        title, icon, updates, type, slug,
        excerpt, body, order, isReference,
        deprecated, hidden, sync_unique, link_url,
        link_external, reusableContent, previousSlug,
        slugUpdatedAt, revision, _id, parentDoc,
        category, project, user, createdAt,
        updatedAt, version, lastUpdatedHash, __v,
        isApi, id, body_html.
    """

    readme_key = get_secrets('README-PUBLIC-DOCS-WEBSITE')['API-KEY']
    url = "https://dash.readme.com/api/v1/docs/{}".format(slug)
    headers = {
        "accept": "application/json",
        "authorization": "Basic {}".format(readme_key)
    }

    response = requests.get(url, headers=headers)
    json_response = json.loads(response.text)
    return json_response
