"""The title of the module description
* The parallel computation utility module

ref: https://research.computing.yale.edu/sites/default/files/files/mpi4py.pdf

TODO:
    * test for each func
    * check the behavior and if it works properly
"""
from typing import Any, Tuple

import numpy as np
from mpi4py import MPI

from src.utils.constants import DirectionIndicators


def Shift(rank_grid: MPI.Cartcomm, direction: int, disp: int) -> Tuple[int, int]:
    """
    Return a tuple (src, dest) of process ranks
    for data shifting with Comm.Sendrecv()

    Args:
        rank_grid (MPI.Cartcomm):

        direction (int):

        disp (int):

    Returns:
        src, dest (Tuple[int, int]):
        src (int):
            The rank of the source process of the information that this process receives.
        dest (int):
            The rank of the destination process of the information that this process sends.

    Note:
        This is a documentation for the MPI.Cartcomm.Shift
        and it is not used in the project.
    """

    return rank_grid.Shift(direction=direction,
                           disp=disp)


def Sendrecv(rank_grid: MPI.Cartcomm, sendbuf: Any, dest: int, sendtag: int,
             recvbuf: Any, source: int, recvtag: int) -> None:
    """
    Send and receive a message

    Args:
        rank_grid (MPI.Cartcomm):
        sendbuf (Any):
            The buffer element for send the information.
        dest (int):
            The rank of the destination process.
        sendtag (int):
            The tag for sending information.
        recvbuf (Any):
            The buffer element for send the information.
        source (int):
            The rank of the source process.
        recvtag (int):
            The tag for receiving information.

    Caution:
        This function is guaranteed not to deadlock in
        situations where pairs of blocking sends and receives may
        deadlock.
        A common mistake when using this function is to
        mismatch the tags with the source and destination ranks,
        which can result in deadlock.

    Note:
        This is a documentation for the MPI.Cartcomm.Shift
        and it is not used in the project.
    """

    rank_grid.Sendrecv(sendbuf=sendbuf, dest=dest, sendtag=sendtag,
                       recvbuf=recvbuf, source=source, recvtag=recvtag)


def ChunkedGridManager():
    def __init__(self, X: int, Y: int):
        self.size = MPI.COMM_WORLD.Get_size()
        self.rank = MPI.COMM_WORLD.Get_rank()
        self.comm = MPI.COMM_WORLD
        self._rank_grid_size = self._compute_rank_grid_size()
        self.rank_grid = self.comm.Create_cart(
            dims=[*self.rank_grid_size],
            periods=[True, True],
            reorder=False
        )
        self.rank_loc = self.rank_grid.Get_coords(self.rank)
        self._local_grid_size = self._compute_local_grid_size(X, Y)
        self._global_grid_size = (X, Y)
        self._x_local_range, self._y_local_range = self._compute_local_range(X, Y)
        self._buffer_grid_size = self._compute_buffer_grid_size()
        self.exist_recvbufer = {}

        # tree structure info
        self.root = bool(self.rank == 0)
        self._compute_tree_structure()                

    @property
    def rank_grid_size(self) -> Tuple[int, int]:
        return self._rank_grid_size

    @property
    def local_grid_size(self) -> Tuple[int, int]:
        return self._local_grid_size

    @property
    def buffer_grid_size(self) -> Tuple[int, int]:
        return self._buffer_grid_size

    @property
    def global_grid_size(self) -> Tuple[int, int]:
        return self._global_grid_size

    @property
    def x_local_range(self) -> Tuple[int, int]:
        return self._x_local_range

    @property
    def y_local_range(self) -> Tuple[int, int]:
        return self._y_local_range

    def _compute_rank_grid_size(self) -> Tuple[int, int]:
        lower, upper = 1, self.size
        for i in range(2, int(np.sqrt(self.size)) + 1):
            if self.size % i == 0:
                lower, upper = self.size // i

        return (lower, upper)

    def _compute_local_grid_size(self, X_global: int, Y_global: int) -> Tuple[int, int]:
        (X_rank, Y_rank) = self.rank_grid_size
        (x_rank, y_rank) = self.rank_loc
        X_small, X_large = X_global // X_rank, (X_global + X_rank - 1) // X_rank
        Y_small, Y_large = Y_global // Y_rank, (Y_global + Y_rank - 1) // Y_rank

        # x_rank, y_rank is 0-indexed
        X_local = X_large if x_rank < X_global % X_rank else X_small
        Y_local = Y_large if y_rank < Y_global % Y_rank else Y_small

        return (X_local, Y_local)

    def _compute_local_range(self, X_global: int, Y_global: int
                             ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """ TODO: Test code """
        (X_rank, Y_rank) = self.rank_grid_size
        (x_rank, y_rank) = self.rank_loc
        (X_local, Y_local) = self._local_grid_size()
        X_small, X_large = X_global // X_rank, (X_global + X_rank - 1) // X_rank
        Y_small, Y_large = Y_global // Y_rank, (Y_global + Y_rank - 1) // Y_rank
        rx, ry = X_global % X_rank, Y_global % Y_rank

        if x_rank < rx:
            x_local_lower = x_rank * X_large
            x_local_upper = x_local_lower + X_large - 1
        else:
            x_local_lower = rx * X_large + (x_rank - rx) * X_small
            x_local_upper = x_local_lower + X_small - 1

        if y_rank < ry:
            y_local_lower = y_rank * Y_large
            y_local_upper = y_local_lower + Y_large - 1
        else:
            y_local_lower = ry * Y_large + (y_rank - ry) * Y_small
            y_local_upper = y_local_lower + Y_small - 1

        """TODO: use it for the sizing of grid size LBM."""
        self.exist_recvbufer[DirectionIndicators.LEFT] = bool(x_local_lower == 0)
        self.exist_recvbufer[DirectionIndicators.RIGHT] = bool(x_local_upper == X_global - 1)
        self.exist_recvbufer[DirectionIndicators.BOTTOM] = bool(y_local_lower == 0)
        self.exist_recvbufer[DirectionIndicators.TOP] = bool(y_local_upper == Y_global - 1)

        return (x_local_lower, x_local_upper), (y_local_lower, y_local_upper)

    def _compute_buffer_grid_size(self) -> Tuple[int, int]:
        gx, gy = self.local_grid_size[0]
        gx += self.exist_recvbufer[DirectionIndicators.LEFT]
        gx += self.exist_recvbufer[DirectionIndicators.RIGHT]
        gy += self.exist_recvbufer[DirectionIndicators.TOP]
        gy += self.exist_recvbufer[DirectionIndicators.BOTTOM]
        return gx, gy

    def _compute_tree_structure(self) -> None:
        """TODO: test"""
        depth = 0
        for d in range(1, 40):
            if self.rank + 2 <= 1 << d:
                depth = d
                break

        n_nodes_prev_depth = 0 if depth == 1 else 1 << (depth - 2)
        n_nodes_cur_depth = 1 << (depth - 1)
        n_nodes_next_depth = 1 << depth

        # the index in the current depth
        idx = self.rank - (n_nodes_cur_depth - 1)
        parent = n_nodes_prev_depth - 1 + (idx >> 1)
        self.parent = parent if parent >= 0 else None
        self.children = [
            n_nodes_next_depth - 1 + (idx << 1) + i
            for i in range(2)
            if n_nodes_next_depth - 1 + (idx << 1) + i < self.size
        ]

    def _step_to_idx(self, step: int, send: bool) -> int:
        assert step == 1 or step == -1
        if send:
            return -2 if step == 1 else 1
        else:  # recv
            return -1 if step == 1 else 0

    def x_in_process(self, x_global: int) -> bool:
        return self.x_local_range[0] <= x_global <= self.x_local_range[1]

    def y_in_process(self, y_global: int) -> bool:
        return self.y_local_range[0] <= y_global <= self.y_local_range[1]

    def location_in_process(self, x_global: int, y_global: int) -> bool:
        return self.x_in_process(x_global) and self.y_in_process(y_global)

    def global_to_local(self, x_global: int, y_global: int) -> Tuple[int, int]:
        assert location_in_process(x_global, y_global)
        return x_global - self.x_local_range[0], y_global - self.y_local_range[0]

    def is_boundary(self, dir: DirectionIndicators) -> bool:
        if not isinstance(dir, DirectionIndicators):
            raise ValueError(f"Args `dir` must be DirectionIndicators type, but got {type(dir)}.")

        if DirectionIndicators.LEFT:
            return self.x_in_process(0)
        if DirectionIndicators.RIGHT:
            return self.x_in_process(self.global_grid_size[0] - 1)
        if DirectionIndicators.BOTTOM:
            return self.y_in_process(0)
        if DirectionIndicators.TOP:
            return self.y_in_process(self.global_grid_size[1] - 1)
