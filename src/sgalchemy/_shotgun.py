"""Define structural protocols for the consumed provider API."""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Literal, Protocol, TypeAlias, runtime_checkable

REQUIRED_SHOTGUN_METHODS = (
    'batch',
    'find',
    'find_one',
    'revive',
    'schema_entity_read',
    'schema_read',
    'summarize',
    'upload_filmstrip_thumbnail',
    'upload_thumbnail',
)
THUMBNAIL_COUPLED_CLEAR_FIELD = 'filmstrip_image'
THUMBNAIL_PRIMARY_FIELD = 'image'
THUMBNAIL_UPLOAD_METHODS: Mapping[
    str,
    Literal['upload_thumbnail', 'upload_filmstrip_thumbnail'],
] = MappingProxyType(
    {
        THUMBNAIL_PRIMARY_FIELD: 'upload_thumbnail',
        THUMBNAIL_COUPLED_CLEAR_FIELD: 'upload_filmstrip_thumbnail',
    }
)
WireRecord: TypeAlias = dict[str, Any]
WireRecords: TypeAlias = list[WireRecord]
BatchItem: TypeAlias = WireRecord | bool
QueryResult: TypeAlias = WireRecords | WireRecord
WireContainer: TypeAlias = dict[Any, Any] | list[Any] | tuple[Any, ...]
WireTraversalContainer: TypeAlias = Mapping[Any, Any] | list[Any] | tuple[Any, ...]


@runtime_checkable
class AsyncShotgunLike(Protocol):
    """Describe the awaited provider surface consumed by sgalchemy.

    Implementations supply awaitable methods with the documented provider
    result shapes. Supplying this protocol to an engine does not transfer
    client ownership or require a close method.

    Attributes:
        base_url: Complete site URL used to identify the supplied connection.
    """

    base_url: str

    async def batch(self, requests: list[WireRecord]) -> list[BatchItem]:
        """Execute create/update/delete requests atomically.

        Args:
            requests: `{'request_type': ..., 'entity_type': ..., ...}` dicts.

        Returns:
            Per-request results, in request order (dict per create/update, bool
            per delete).
        """
        ...

    async def find(  # noqa: PLR0913, PLR0917 - Preserve provider call compatibility.
        self,
        entity_type: str,
        filters: list[Any],
        fields: list[str] | None = None,
        order: list[dict[str, str]] | None = None,
        filter_operator: str | None = None,
        limit: int = 0,
        retired_only: bool = False,
        page: int = 0,
        include_archived_projects: bool = True,
        additional_filter_presets: list[dict[str, Any]] | None = None,
    ) -> WireRecords:
        """Return entity records matching the filters.

        Args:
            entity_type: Entity type name to query.
            filters: Wire-format filter list.
            fields: Field codes to return (server defaults to `id`).
            order: `{'field_name': ..., 'direction': ...}` sort items.
            filter_operator: `'all'` or `'any'` across top-level filters.
            limit: Maximum records to return; 0 means no limit.
            retired_only: Return only retired entities.
            page: 1-based page of results to return; 0 means all.
            include_archived_projects: Include entities of archived projects.
            additional_filter_presets: Server-side filter presets, ANDed in.

        Returns:
            One `{'type': ..., 'id': ..., <field>: ...}` dict per record.
        """
        ...

    async def find_one(
        self,
        entity_type: str,
        filters: list[Any],
        fields: list[str] | None = None,
    ) -> WireRecord | None:
        """Return the first matching record, or None.

        Args:
            entity_type: Entity type name to query.
            filters: Wire-format filter list.
            fields: Field codes to return (server defaults to `id`).

        Returns:
            The first matching record dict, or None when nothing matches.
        """
        ...

    async def revive(self, entity_type: str, entity_id: int) -> bool:
        """Restore a retired entity.

        Args:
            entity_type: Entity type name.
            entity_id: Id of the retired entity.

        Returns:
            Whether the entity was revived.
        """
        ...

    async def schema_entity_read(self) -> dict[str, dict[str, Any]]:
        """Return entity-type metadata (display names, visibility).

        Returns:
            Mapping of entity type name to its metadata dict.
        """
        ...

    async def schema_read(self) -> dict[str, dict[str, Any]]:
        """Return the field schema for every entity type.

        Returns:
            Mapping of entity type name to its field-schema dicts.
        """
        ...

    async def summarize(  # noqa: PLR0913, PLR0917 - Preserve provider call compatibility.
        self,
        entity_type: str,
        filters: list[Any],
        summary_fields: list[dict[str, str]],
        filter_operator: str | None = None,
        grouping: list[dict[str, str]] | None = None,
        include_archived_projects: bool = True,
    ) -> WireRecord:
        """Return aggregate values and optional grouping rows.

        Args:
            entity_type: Entity type name to summarize.
            filters: Wire-format filter list.
            summary_fields: Summary field descriptors.
            filter_operator: `all` or `any` across top-level filters.
            grouping: Optional grouping descriptors.
            include_archived_projects: Include entities of archived projects.

        Returns:
            Summary and grouping payload from the provider.
        """
        ...

    async def upload_filmstrip_thumbnail(
        self,
        entity_type: str,
        entity_id: int,
        path: str,
    ) -> int:
        """Upload and assign a filmstrip thumbnail.

        Args:
            entity_type: Entity type name.
            entity_id: Id of the entity receiving the filmstrip.
            path: Local path of the filmstrip image file.

        Returns:
            Id of the created attachment.
        """
        ...

    async def upload_thumbnail(
        self,
        entity_type: str,
        entity_id: int,
        path: str,
    ) -> int:
        """Upload and assign a thumbnail.

        Args:
            entity_type: Entity type name.
            entity_id: Id of the entity receiving the thumbnail.
            path: Local path of the image file.

        Returns:
            Id of the created attachment.
        """
        ...


@runtime_checkable
class AsyncClosableShotgunLike(AsyncShotgunLike, Protocol):
    """Async call surface with provider-owned lifecycle support."""

    async def close(self) -> None:
        """Close provider-owned async resources."""
        ...


class ShotgunLike(Protocol):
    """Describe the synchronous provider surface consumed by sgalchemy.

    Implementations supply synchronous methods with the documented provider
    result shapes. Supplying this protocol to an engine does not transfer
    client ownership or require a close method.

    Attributes:
        base_url: Complete site URL used to identify the supplied connection.
    """

    base_url: str

    def batch(self, requests: list[WireRecord]) -> list[BatchItem]:
        """Execute create/update/delete requests atomically.

        Args:
            requests: `{'request_type': ..., 'entity_type': ..., ...}` dicts.

        Returns:
            Per-request results, in request order (dict per create/update, bool
            per delete).
        """
        ...

    def find(  # noqa: PLR0913, PLR0917 - Preserve provider call compatibility.
        self,
        entity_type: str,
        filters: list[Any],
        fields: list[str] | None = None,
        order: list[dict[str, str]] | None = None,
        filter_operator: str | None = None,
        limit: int = 0,
        retired_only: bool = False,
        page: int = 0,
        include_archived_projects: bool = True,
        additional_filter_presets: list[dict[str, Any]] | None = None,
    ) -> WireRecords:
        """Return entity records matching the filters.

        Args:
            entity_type: Entity type name to query.
            filters: Wire-format filter list.
            fields: Field codes to return (server defaults to `id`).
            order: `{'field_name': ..., 'direction': ...}` sort items.
            filter_operator: `'all'` or `'any'` across top-level filters.
            limit: Maximum records to return; 0 means no limit.
            retired_only: Return only retired entities.
            page: 1-based page of results to return; 0 means all.
            include_archived_projects: Include entities of archived projects.
            additional_filter_presets: Server-side filter presets, ANDed in.

        Returns:
            One `{'type': ..., 'id': ..., <field>: ...}` dict per record.
        """
        ...

    def find_one(
        self,
        entity_type: str,
        filters: list[Any],
        fields: list[str] | None = None,
    ) -> WireRecord | None:
        """Return the first matching record, or None.

        Args:
            entity_type: Entity type name to query.
            filters: Wire-format filter list.
            fields: Field codes to return (server defaults to `id`).

        Returns:
            The first matching record dict, or None when nothing matches.
        """
        ...

    def revive(self, entity_type: str, entity_id: int) -> bool:
        """Restore a retired entity.

        Args:
            entity_type: Entity type name.
            entity_id: Id of the retired entity.

        Returns:
            Whether the entity was revived.
        """
        ...

    def schema_entity_read(self) -> dict[str, dict[str, Any]]:
        """Return entity-type metadata (display names, visibility).

        Returns:
            Mapping of entity type name to its metadata dict.
        """
        ...

    def schema_read(self) -> dict[str, dict[str, Any]]:
        """Return the field schema for every entity type.

        Returns:
            Mapping of entity type name to its field-schema dicts.
        """
        ...

    def summarize(  # noqa: PLR0913, PLR0917 - Preserve provider call compatibility.
        self,
        entity_type: str,
        filters: list[Any],
        summary_fields: list[dict[str, str]],
        filter_operator: str | None = None,
        grouping: list[dict[str, str]] | None = None,
        include_archived_projects: bool = True,
    ) -> WireRecord:
        """Return aggregate values and optional grouping rows.

        Args:
            entity_type: Entity type name to summarize.
            filters: Wire-format filter list.
            summary_fields: Summary field descriptors.
            filter_operator: `all` or `any` across top-level filters.
            grouping: Optional grouping descriptors.
            include_archived_projects: Include entities of archived projects.

        Returns:
            Summary and grouping payload from the provider.
        """
        ...

    def upload_filmstrip_thumbnail(self, entity_type: str, entity_id: int, path: str) -> int:
        """Upload and assign a filmstrip thumbnail.

        Args:
            entity_type: Entity type name.
            entity_id: Id of the entity receiving the filmstrip.
            path: Local path of the filmstrip image file.

        Returns:
            Id of the created attachment.
        """
        ...

    def upload_thumbnail(self, entity_type: str, entity_id: int, path: str) -> int:
        """Upload and assign a thumbnail.

        Args:
            entity_type: Entity type name.
            entity_id: Id of the entity receiving the thumbnail.
            path: Local path of the image file.

        Returns:
            Id of the created attachment.
        """
        ...


class ClosableShotgunLike(ShotgunLike, Protocol):
    """Sync call surface with provider-owned lifecycle support."""

    def close(self) -> None:
        """Close provider-owned resources."""
        ...
