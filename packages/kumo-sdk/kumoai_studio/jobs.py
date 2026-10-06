# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from typing import Generic, Mapping, Optional, TypeVar

from kumoapi.jobs import JobStatusReport
from typing_extensions import Self

from kumoai_studio.backend import JobBackend
from kumoai_studio.client.jobs import JobRequestType, JobResourceType

IDType = TypeVar('IDType', bound=str)


class JobInterface(ABC, Generic[IDType, JobRequestType, JobResourceType]):
    r"""Defines a standard interface for job objects."""

    _kumo_project_id: str

    @abstractmethod
    def __init__(self, job_id: IDType,
                 project_id: Optional[str] = None) -> None:
        pass

    @staticmethod
    @abstractmethod
    def _api() -> JobBackend:
        pass

    @classmethod
    def search_by_tags(
        cls,
        tags: Mapping[str, str],
        limit: int = 10,
        project_id: Optional[str] = None,
    ) -> list[Self]:
        r"""Returns a list of job instances from a set of job tags.

        Args:
            tags (Mapping[str, str]): Tags by which to search.
            limit (int): Max number of jobs to list, default 10.
            project_id: Exact project to search. ``None`` uses the active
                project. Every returned job snapshots the project from its
                server resource.

        Example:
            >>> # doctest: +SKIP
            >>> tags = {'pquery_name': 'my_pquery_name'}
            >>> jobs = BatchPredictionJob.search_by_tags(tags)
            Search limited to 10 results based on the `limit` parameter.
            Found 2 jobs.
        """
        print(f"Search limited to {limit} results based on the `limit` "
              "parameter.")

        jobs = cls._api().list(limit=limit, additional_tags=tags,
                               project_id=project_id)

        print(f"Found {len(jobs)} jobs.")

        return [cls(j.job_id, project_id=j.project_id) for j in jobs]

    @property
    @abstractmethod
    def id(self) -> IDType:
        pass

    @abstractmethod
    def status(self) -> JobStatusReport:
        pass

    def get_tags(self) -> dict[str, str]:
        r"""Returns the tags of the job."""
        return self._api().get(self.id, project_id=self._kumo_project_id).tags

    def delete_tags(self, tags: list[str]) -> bool:
        r"""Removes the tags from the job.

        Args:
            tags (list[str]): The tags to remove.
        """
        return self._api().delete_tags(self.id, tags,
                                       project_id=self._kumo_project_id)

    def update_tags(self, tags: Mapping[str, Optional[str]]) -> bool:
        r"""Updates the tags of the job.

        Args:
            tags (Mapping[str, Optional[str]]): The tags to update.
                Note that the value 'none' will remove the tag. If the tag is
                not present, it will be added.
        """
        return self._api().update_tags(self.id, tags,
                                       project_id=self._kumo_project_id)

    @abstractmethod
    def load_config(self) -> JobRequestType:
        pass

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(job_id={self.id})'
