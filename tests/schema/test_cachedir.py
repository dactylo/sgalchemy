from pathlib import Path
from typing import cast

import pytest

from sgalchemy.schema import _cachedir


def test_cache_file_key_includes_site_scope_locale(tmp_path: Path) -> None:
    p = _cachedir.cache_file('https://studio.shotgunstudio.com', root=tmp_path)
    assert p.name.startswith('studio.shotgunstudio.com__global__none__')
    q = _cachedir.cache_file(
        'https://studio.shotgunstudio.com', scope='project-70', locale='ja', root=tmp_path
    )
    assert q.name.startswith('studio.shotgunstudio.com__project-70__ja__')
    assert p != q


def test_cache_file_rejects_incomplete_site_with_corrective_message(tmp_path: Path) -> None:
    match = r'provide a complete URL with both scheme and hostname.*https://studio\.example\.com'
    with pytest.raises(ValueError, match=match):
        _cachedir.cache_file('studio.example.com', root=tmp_path)


def test_cache_file_uses_complete_url_and_windows_safe_name(tmp_path: Path) -> None:
    sites = (
        'https://studio.example.com',
        'http://studio.example.com',
        'https://studio.example.com:8443',
        'https://studio.example.com/tenant-a',
    )
    paths = [_cachedir.cache_file(site, scope='project/70', root=tmp_path) for site in sites]
    assert len(set(paths)) == len(sites)
    assert all(path.parent == tmp_path for path in paths)
    assert all(not set('<>:"/\\|?*') & set(path.name) for path in paths)
    assert _cachedir.cache_file(f'{sites[0]}/', root=tmp_path) == _cachedir.cache_file(
        sites[0], root=tmp_path
    )


def test_env_var_overrides_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('SGALCHEMY_SCHEMA_CACHE', str(tmp_path))
    p = _cachedir.cache_file('https://x.example.com')
    assert p.parent == tmp_path


@pytest.mark.parametrize(
    ('platform', 'local_app_data', 'xdg_cache_home', 'expected'),
    [
        ('win32', 'C:/LocalData', None, Path('C:/LocalData') / 'sgalchemy' / 'Cache'),
        (
            'win32',
            None,
            None,
            Path('/home/test-user') / 'AppData' / 'Local' / 'sgalchemy' / 'Cache',
        ),
        (
            'win32',
            '',
            None,
            Path('/home/test-user') / 'AppData' / 'Local' / 'sgalchemy' / 'Cache',
        ),
        (
            'darwin',
            '/ignored/local',
            '/ignored/xdg',
            Path('/home/test-user/Library/Caches/sgalchemy'),
        ),
        (
            'linux',
            None,
            str(Path('/xdg/cache').absolute()),
            Path('/xdg/cache').absolute() / 'sgalchemy',
        ),
        ('linux', None, None, Path('/home/test-user/.cache/sgalchemy')),
        # The XDG Base Directory specification treats empty and relative values as unset.
        ('linux', None, '', Path('/home/test-user/.cache/sgalchemy')),
        ('linux', None, 'relative/cache', Path('/home/test-user/.cache/sgalchemy')),
    ],
)
def test_user_cache_dir_uses_platform_path(
    platform: str,
    local_app_data: str | None,
    xdg_cache_home: str | None,
    expected: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = Path('/home/test-user')
    monkeypatch.setattr(_cachedir.sys, 'platform', platform)
    monkeypatch.setattr(_cachedir.Path, 'home', cast('object', classmethod(lambda _cls: home)))
    for name, value in (
        ('LOCALAPPDATA', local_app_data),
        ('XDG_CACHE_HOME', xdg_cache_home),
    ):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)

    assert _cachedir._user_cache_dir() == expected
