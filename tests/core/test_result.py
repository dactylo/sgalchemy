import copy
import pickle
from typing import TypeVar

import pytest

from sgalchemy._core._result import Result, Row

_T = TypeVar('_T')


def _pickle_round_trip(value: _T) -> _T:
    payload = pickle.dumps(value)
    deserialize = pickle.loads
    return deserialize(payload)


def test_result_copy_deepcopy_and_pickle_round_trip() -> None:
    row = Row({'code': 'A'})
    result = Result([row])

    shallow = copy.copy(result)
    deep = copy.deepcopy(result)
    round_trip = _pickle_round_trip(result)

    assert type(shallow) is Result
    assert shallow == result
    assert shallow is not result
    assert shallow[0] is row
    for restored in (deep, round_trip):
        assert type(restored) is Result
        assert restored == result
        assert restored is not result
        assert type(restored[0]) is Row
        assert restored[0] is not row


def test_result_is_list_like() -> None:
    row = Row({'code': 'A'})
    result = Result([row])
    assert result[0] is row
    assert list(result) == [row]


def test_row_copies_values_and_maps_attribute_access() -> None:
    source = {'code': 'A'}
    row = Row(source)
    source['code'] = 'B'
    assert row['code'] == 'A'
    assert row.code == 'A'
    assert list(row) == ['code']
    assert len(row) == 1
    with pytest.raises(KeyError):
        _ = row['missing']
    with pytest.raises(AttributeError, match='missing'):
        _ = row.missing


def test_row_copy_deepcopy_and_pickle_round_trip() -> None:
    row = Row({'code': 'A'})

    for restored in (
        copy.copy(row),
        copy.deepcopy(row),
        _pickle_round_trip(row),
    ):
        assert type(restored) is Row
        assert restored == row
        assert restored is not row
