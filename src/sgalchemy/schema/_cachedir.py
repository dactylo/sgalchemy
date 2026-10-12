"""Platform cache directory and cache key."""

import hashlib
import os
import re
import sys
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

_ENV_VAR = 'SGALCHEMY_SCHEMA_CACHE'


def _normalize_site(site: str) -> tuple[str, str]:
    """Return a normalized complete URL and readable hostname."""
    parsed = urlsplit(site)
    if not parsed.scheme or not parsed.netloc or parsed.hostname is None:
        raise ValueError(
            f'invalid site URL {site!r}; provide a complete URL with both scheme and hostname, '
            "for example 'https://studio.example.com'"
        )
    normalized = urlunsplit(
        (
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            parsed.path.rstrip('/'),
            parsed.query,
            parsed.fragment,
        )
    )
    return normalized, parsed.hostname.lower()


def _safe_component(value: str) -> str:
    """Return one readable filename component without platform-reserved characters."""
    return re.sub(r'[^A-Za-z0-9._-]+', '-', value).strip('._-') or 'none'


def _user_cache_dir() -> Path:
    """Return the platform-appropriate per-user cache directory."""
    if sys.platform == 'win32':
        local_app_data = os.environ.get('LOCALAPPDATA')
        base = Path(local_app_data) if local_app_data else Path.home() / 'AppData' / 'Local'
        return base / 'sgalchemy' / 'Cache'
    if sys.platform == 'darwin':
        return Path.home() / 'Library' / 'Caches' / 'sgalchemy'
    # The XDG Base Directory specification treats an empty or relative value as unset.
    xdg_cache_home = Path(os.environ.get('XDG_CACHE_HOME', ''))
    base = xdg_cache_home if xdg_cache_home.is_absolute() else Path.home() / '.cache'
    return base / 'sgalchemy'


def cache_file(
    site: str, *, scope: str = 'global', locale: str = 'none', root: Path | None = None
) -> Path:
    """Compute a schema cache path keyed by site, scope, and locale.

    This function creates no files or directories.

    Args:
        site: Complete site URL.
        scope: Schema-scope identity included in the cache key.
        locale: Schema-locale identity included in the cache key.
        root: Cache directory override. `None` uses a nonempty
            `SGALCHEMY_SCHEMA_CACHE`, otherwise the platform user-cache
            directory.

    Returns:
        Cache path containing readable components and a digest of the
        normalized site, scope, and locale.

    Raises:
        ValueError: If `site` is not a complete URL.
    """
    if root is None:
        env = os.environ.get(_ENV_VAR)
        root = Path(env) if env else _user_cache_dir()
    normalized, host = _normalize_site(site)
    digest = hashlib.sha256(f'{normalized}\0{scope}\0{locale}'.encode()).hexdigest()[:20]
    return root / (
        f'{_safe_component(host)}__{_safe_component(scope)}__{_safe_component(locale)}__'
        f'{digest}.json'
    )
