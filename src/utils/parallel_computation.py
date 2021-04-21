"""The title of the module description
* The parallel computation utility module

ref: https://research.computing.yale.edu/sites/default/files/files/mpi4py.pdf

TODO:
    * TODO1
    * TODO2
"""
from typing import List, Tuple

import numpy as np
from mpi4py import MPI

from src.utils.constants import DirectionIndicators, DIRECTION2VEC


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


def Sendrecv(rank_grid: MPI.Cartcomm) -> None:
    """
    Send and receive a message

    Args:
        rank_grid (MPI.Cartcomm):
        sendbuf (Any):
            The buffer element for send the information.
        dest (int):
            The rank of the destination process.
        recvtag (int):
            The tag for receiving information.
        recvbuf (Any):
            The buffer element for send the information.
        source (int):
            The rank of the source process.
        sendtag (int):
            The tag for sending information.
        status (Optional[Status]):
            status.

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

    rank_grid.Sendrecv()


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
        self._neighbor_proc_locations = self._compute_neighbor_proc_locations()

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

        return (x_local_lower, x_local_upper), (y_local_lower, y_local_upper)

    def _compute_neighbor_proc_locations(self) -> List[DirectionIndicators]:
        X, Y = self.global_grid_size
        neighbor_proc_locations = []
        if not self.x_in_process(0):
            neighbor_proc_locations.append(DirectionIndicators.LEFT)
        if not self.x_in_process(X - 1):
            neighbor_proc_locations.append(DirectionIndicators.RIGHT)
        if not self.y_in_process(0):
            neighbor_proc_locations.append(DirectionIndicators.BOTTOM)
        if not self.y_in_process(Y - 1):
            neighbor_proc_locations.append(DirectionIndicators.TOP)

        return neighbor_proc_locations

    def communicate_with_neighbors(self) -> None:
        x_rank, y_rank = self.rank_loc

        for dir in self.neighbor_proc_locations:
            dx, dy = DIRECTION2VEC[dir]
            src, dst = self.rank_loc.Shift(direction=int(dx == 0), disp=(dy if dx == 0 else dy))
            x_rank + dx, y_rank + dy
            recvbuf = """"""
            self.rank_loc.Sendrecv("""""", dst, recvbuf=recvbuf, source=src)
            # array = recvbuf

    @property
    def rank_grid_size(self) -> Tuple[int, int]:
        return self._rank_grid_size

    @property
    def local_grid_size(self) -> Tuple[int, int]:
        return self._local_grid_size

    @property
    def global_grid_size(self) -> Tuple[int, int]:
        return self._global_grid_size

    @property
    def x_local_range(self) -> Tuple[int, int]:
        return self._x_local_range

    @property
    def y_local_range(self) -> Tuple[int, int]:
        return self._y_local_range

    @property
    def neighbor_proc_locations(self) -> List[DirectionIndicators]:
        return self._neighbor_proc_locations

    def x_in_process(self, x_global: int) -> bool:
        return self.x_local_range[0] <= x_global <= self.x_local_range[1]

    def y_in_process(self, y_global: int) -> bool:
        return self.y_local_range[0] <= y_global <= self.y_local_range[1]

    def location_in_process(self, x_global: int, y_global: int) -> bool:
        return self.x_in_process(x_global) and self.y_in_process(y_global)

    def global_to_local(self, x_global: int, y_global: int) -> Tuple[int, int]:
        assert location_in_process(x_global, y_global)
        return x_global - self.x_local_range[0], y_global - self.y_local_range[0]
