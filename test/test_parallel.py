import unittest

from src.utils.parallel_computation import ChunkedGridManager
from src.utils.constants import DirectionIndicators


class TestChunkedGridManager(unittest.TestCase):
    def setUp(self) -> None:
        self.ranks = list(range(8))
        self.X, self.Y = (5, 6)
        self.rank_locs = [(0, 0), (0, 1), (0, 2), (0, 3),
                          (1, 0), (1, 1), (1, 2), (1, 3)]

    def test_compute_rank_grid_size(self) -> None:
        cgm = ChunkedGridManager(self.X, self.Y)
        cgm._size = 8
        assert cgm._compute_rank_grid_size(2, 4) == (2, 4)
        assert cgm._compute_rank_grid_size(4, 2) == (4, 2)
        cgm._size = 42
        assert cgm._compute_rank_grid_size(2, 4) == (6, 7)
        assert cgm._compute_rank_grid_size(4, 2) == (7, 6)

    def test_compute_local_grid_size(self) -> None:
        cgm = ChunkedGridManager(self.X, self.Y)
        cgm._rank_grid_size = (2, 4)
        anses = [(3, 2), (3, 2), (3, 1), (3, 1),
                 (2, 2), (2, 2), (2, 1), (2, 1)]

        for ans, rank_loc in zip(anses, self.rank_locs):
            cgm._rank_loc = rank_loc
            val = cgm._compute_local_grid_size(self.X, self.Y)
            assert val == ans

    def test_compute_local_range(self) -> None:
        cgm = ChunkedGridManager(self.X, self.Y)
        cgm._rank_grid_size = (2, 4)
        anses = [((0, 2), (0, 1)), ((0, 2), (2, 3)),
                 ((0, 2), (4, 4)), ((0, 2), (5, 5)),
                 ((3, 4), (0, 1)), ((3, 4), (2, 3)),
                 ((3, 4), (4, 4)), ((3, 4), (5, 5))]

        for ans, rank_loc in zip(anses, self.rank_locs):
            cgm._rank_loc = rank_loc
            cgm._local_grid_size = cgm._compute_local_grid_size(self.X, self.Y)
            val = cgm._compute_local_range(self.X, self.Y)
            assert val == ans

    def test_compute_buffer_grid_size(self) -> None:
        cgm = ChunkedGridManager(self.X, self.Y)
        dx, dy = cgm.local_grid_size
        val = cgm._compute_buffer_grid_size()
        assert val == (dx + 2, dy + 2)

    def test_compute_tree_structure(self) -> None:
        cgm = ChunkedGridManager(self.X, self.Y)
        cgm._rank = 4
        cgm._size = 20
        childrens = [[9, 10], [15, 16], [1, 2],
                     [19], []]
        parents = [1, 3, None, 4, 4]
        ranks = [4, 7, 0, 9, 10]

        for c, p, r in zip(childrens, parents, ranks):
            cgm._rank = r
            cgm._compute_tree_structure()
            assert cgm.parent == p
            assert cgm.children == c

    def test_global_to_local(self) -> None:
        cgm = ChunkedGridManager(self.X, self.Y)
        cgm._x_local_range = (2, 4)
        cgm._y_local_range = (3, 4)
        val = cgm.global_to_local(2, 3)
        assert val == (0, 0)
        val = cgm.global_to_local(2, 4)
        assert val == (0, 1)
        val = cgm.global_to_local(3, 3)
        assert val == (1, 0)
        val = cgm.global_to_local(3, 4)
        assert val == (1, 1)
        val = cgm.global_to_local(4, 3)
        assert val == (2, 0)
        val = cgm.global_to_local(4, 4)
        assert val == (2, 1)

    def test_is_boundary(self) -> None:
        cgm = ChunkedGridManager(self.X, self.Y)
        cgm._x_local_range = (0, 2)
        cgm._y_local_range = (0, 3)
        for dir in DirectionIndicators:
            ans = (DirectionIndicators.LEFT == dir
                   or DirectionIndicators.BOTTOM == dir)
            val = cgm.is_boundary(dir)
            assert ans == val


if __name__ == '__main__':
    unittest.main()
