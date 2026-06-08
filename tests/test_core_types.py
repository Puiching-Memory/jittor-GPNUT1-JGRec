import numpy as np
import pytest

from jgrec.core.types import InteractionTable


def test_interaction_table_rejects_integer_row_indexing():
    table = InteractionTable.from_array(
        np.asarray(
            [
                [1, 10, 100],
                [2, 20, 200],
            ],
            dtype=np.int32,
        )
    )

    with pytest.raises(TypeError, match="does not support integer row indexing"):
        table[0]


def test_interaction_table_supports_table_slices_and_take():
    table = InteractionTable.from_array(
        np.asarray(
            [
                [1, 10, 100],
                [2, 20, 200],
                [3, 30, 300],
            ],
            dtype=np.int32,
        )
    )

    sliced = table[1:]
    taken = table.take(np.asarray([2, 0], dtype=np.int64))

    np.testing.assert_array_equal(sliced.to_array(), np.asarray([[2, 20, 200], [3, 30, 300]], dtype=np.int32))
    np.testing.assert_array_equal(taken.to_array(), np.asarray([[3, 30, 300], [1, 10, 100]], dtype=np.int32))
