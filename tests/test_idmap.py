from jgrec.core.types import Interaction
from jgrec.idmap import NodeIdMap


def test_node_id_map_assigns_sorted_dense_ids():
    id_map = NodeIdMap.from_interactions(
        [
            Interaction(src=20, dst=300, time=1),
            Interaction(src=10, dst=100, time=2),
            Interaction(src=20, dst=200, time=3),
        ]
    )

    assert id_map.src_values == (10, 20)
    assert id_map.dst_values == (100, 200, 300)
    assert id_map.num_src == 2
    assert id_map.num_dst == 3
    assert id_map.src_id(20) == 1
    assert id_map.src_id(99) == -1
    assert id_map.dst_ids((300, 99, 100)).tolist() == [2, -1, 0]
