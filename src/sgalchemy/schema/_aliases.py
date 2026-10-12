"""Derive Python identifiers that alias schema names."""

import keyword
import re
from collections.abc import Collection, Mapping

from sgalchemy.schema._model import EntitySchema

_NON_IDENTIFIER_CHARACTERS = re.compile(r'[^0-9A-Za-z]+')


def _capwords_identifier(display_name: str) -> str | None:
    """Sanitize a display name to a CapWords identifier when possible."""
    parts = [part for part in _NON_IDENTIFIER_CHARACTERS.split(display_name) if part]
    if not parts:
        return None
    candidate = ''.join(part[:1].upper() + part[1:] for part in parts)
    return (
        candidate
        if candidate.isidentifier()
        and not candidate[0].isdigit()
        and not keyword.iskeyword(candidate)
        else None
    )


def entity_aliases(
    schemas: Mapping[str, EntitySchema],
) -> tuple[dict[str, str], list[str]]:
    """Derive display-name aliases for entity types.

    Args:
        schemas: Parsed schemas keyed by canonical type name.

    Returns:
        The aliases keyed by identifier and human-readable collision reports.
    """
    aliases: dict[str, str] = {}
    reports: list[str] = []
    canonical_names = set(schemas)
    for type_name, entity in sorted(schemas.items()):
        alias = _capwords_identifier(entity.display_name)
        if alias is None or alias == type_name:
            continue
        if alias in canonical_names:
            reports.append(f'entity alias {alias!r} for {type_name} shadows a type; skipped')
        elif alias in aliases:
            reports.append(
                f'entity alias {alias!r} for {type_name} collides with {aliases[alias]}; skipped'
            )
        else:
            aliases[alias] = type_name
    return aliases, reports


def field_aliases(
    entity: EntitySchema,
    *,
    reserved_names: Collection[str] = (),
) -> tuple[dict[str, str], list[str]]:
    """Derive aliases by stripping `sg_` from an entity's field codes.

    Args:
        entity: The entity schema from which to derive aliases.
        reserved_names: Attribute names already owned by the entity API.

    Returns:
        The aliases keyed by identifier and human-readable collision reports.
    """
    aliases: dict[str, str] = {}
    reports: list[str] = []
    for code in sorted(entity.fields):
        if not code.startswith('sg_'):
            continue
        alias = code[3:]
        if not alias.isidentifier():
            continue
        if keyword.iskeyword(alias) or alias.startswith('_') or alias in reserved_names:
            reports.append(
                f'{entity.type_name}: field alias {alias!r} for {code} shadows '
                'Python or Entity API; skipped'
            )
        elif alias in entity.fields or alias in aliases:
            reports.append(
                f'{entity.type_name}: field alias {alias!r} for {code} collides; skipped'
            )
        else:
            aliases[alias] = code
    return aliases, reports
