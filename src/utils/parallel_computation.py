"""The title of the module description
* The parallel computation utility module

ref: https://research.computing.yale.edu/sites/default/files/files/mpi4py.pdf

TODO:
    * None
"""
from typing import Any, List, Tuple, Union

import numpy as np
from numpy.lib.format import dtype_to_descr, magic
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


class ChunkedGridManager():
    def __init__(self, X: int, Y: int):
        self._size = MPI.COMM_WORLD.Get_size()
        self._rank = MPI.COMM_WORLD.Get_rank()
        self._comm = MPI.COMM_WORLD
        self._rank_grid_size = self._compute_rank_grid_size(X, Y)
        self._rank_grid = self.comm.Create_cart(
            dims=[*self.rank_grid_size],
            periods=[True, True],
            reorder=False
        )
        self._rank_loc = self.rank_grid.Get_coords(self.rank)
        self._local_grid_size = self._compute_local_grid_size(X, Y)
        self._global_grid_size = (X, Y)
        self._x_local_range, self._y_local_range = self._compute_local_range(X, Y)
        self._buffer_grid_size = self._compute_buffer_grid_size()
        self._neighbor_directions = self._compute_neighbor_directions()

    @property
    def rank_grid_size(self) -> Tuple[int, int]:
        return self._rank_grid_size

    @property
    def rank_loc(self) -> Tuple[int, int]:
        return self._rank_loc

    @property
    def rank_grid(self) -> MPI.Cartcomm:
        return self._rank_grid

    @property
    def size(self) -> int:
        return self._size

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def comm(self) -> MPI.Intracomm:
        return self._comm

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

    @property
    def neighbor_directions(self) -> List[DirectionIndicators]:
        return self._neighbor_directions

    def _compute_rank_grid_size(self, X_global: int, Y_global: int) -> Tuple[int, int]:
        lower, upper = 1, self.size
        for i in range(2, int(np.sqrt(self.size)) + 1):
            if self.size % i == 0:
                lower, upper = i, self.size // i

        return (lower, upper) if X_global <= Y_global else (upper, lower)

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

        (X_rank, Y_rank) = self.rank_grid_size
        (x_rank, y_rank) = self.rank_loc
        (X_local, Y_local) = self.local_grid_size
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

    def _compute_neighbor_directions(self) -> List[DirectionIndicators]:
        return [dir for dir in DirectionIndicators if self.exist_neighbor(dir)]

    def _compute_buffer_grid_size(self) -> Tuple[int, int]:
        gx, gy = self.local_grid_size
        x_start, y_start = 0, 0
        x_end, y_end = self.local_grid_size
        if not self.is_boundary(DirectionIndicators.LEFT):
            gx += 1
            x_start += 1
            x_end += 1
        if not self.is_boundary(DirectionIndicators.RIGHT):
            gx += 1
        if not self.is_boundary(DirectionIndicators.BOTTOM):
            gy += 1
            y_start += 1
            y_end += 1
        if not self.is_boundary(DirectionIndicators.TOP):
            gy += 1

        self.x_valid_slice = (x_start, x_end)
        self.y_valid_slice = (y_start, y_end)
        return (gx, gy)

    def _step_to_idx(self, dx: int, dy: int, send: bool) -> Union[Tuple[int, int], int]:
        assert dx != 0 or dy != 0
        if send:
            x_nxt = -2 if dx == 1 else 1
            y_nxt = -2 if dy == 1 else 1
        else:
            x_nxt = -1 if dx == 1 else 0
            y_nxt = -1 if dy == 1 else 0

        if dx == 0 or dy == 0:
            return x_nxt if dx != 0 else y_nxt
        else:
            return (x_nxt, y_nxt)

    def x_in_process(self, x_global: int) -> bool:
        return self.x_local_range[0] <= x_global <= self.x_local_range[1]

    def y_in_process(self, y_global: int) -> bool:
        return self.y_local_range[0] <= y_global <= self.y_local_range[1]

    def location_in_process(self, x_global: int, y_global: int) -> bool:
        return self.x_in_process(x_global) and self.y_in_process(y_global)

    def global_to_local(self, x_global: int, y_global: int) -> Tuple[int, int]:
        assert self.location_in_process(x_global, y_global)
        return x_global - self.x_local_range[0], y_global - self.y_local_range[0]

    def is_boundary(self, dir: DirectionIndicators) -> bool:
        if not isinstance(dir, DirectionIndicators):
            raise ValueError(f"Args `dir` must be DirectionIndicators type, but got {type(dir)}.")

        X_global, Y_global = self.global_grid_size

        if dir == DirectionIndicators.LEFT:
            return self.x_in_process(0)
        elif dir == DirectionIndicators.RIGHT:
            return self.x_in_process(X_global - 1)
        elif dir == DirectionIndicators.BOTTOM:
            return self.y_in_process(0)
        elif dir == DirectionIndicators.TOP:
            return self.y_in_process(Y_global - 1)
        else:
            raise ValueError("dir must be either {TOP, BOTTOM, LEFT, RIGHT}.")

    def exist_neighbor(self, dir: DirectionIndicators) -> bool:
        if not isinstance(dir, DirectionIndicators):
            raise ValueError(f"Args `dir` must be DirectionIndicators type, but got {type(dir)}.")

        X_global, Y_global = self.global_grid_size
        if dir in [getattr(DirectionIndicators, s_dir) for s_dir in ['TOP', 'BOTTOM', 'LEFT', 'RIGHT']]:
            return not self.is_boundary(dir)
        elif dir == DirectionIndicators.RIGHTTOP:
            return not self.is_boundary(DirectionIndicators.RIGHT) and not self.is_boundary(DirectionIndicators.TOP)
        elif dir == DirectionIndicators.LEFTTOP:
            return not self.is_boundary(DirectionIndicators.LEFT) and not self.is_boundary(DirectionIndicators.TOP)
        elif dir == DirectionIndicators.RIGHTBOTTOM:
            return not self.is_boundary(DirectionIndicators.RIGHT) and not self.is_boundary(DirectionIndicators.BOTTOM)
        elif dir == DirectionIndicators.LEFTBOTTOM:
            return not self.is_boundary(DirectionIndicators.LEFT) and not self.is_boundary(DirectionIndicators.BOTTOM)
        else:
            raise ValueError("dir must be either {TOP, BOTTOM, LEFT, RIGHT, "
                             "RIGHTTOP, LEFTTOP, RIGHTBOTTOM, LEFTBOTTOM}.")

    def get_neighbor_rank(self, dir: DirectionIndicators) -> int:
        assert self.exist_neighbor(dir)
        dx, dy = DIRECTION2VEC[dir]
        rX, rY = self.rank_grid_size
        rx, ry = self.rank_loc
        rx, ry = (rx + dx + rX) % rX, (ry + dy + rY) % rY
        return rx * rY + ry

    def save_mpiio(self, file_name: str, vec: np.ndarray) -> None:
        """
        Write a global two-dimensional array to a single file in the npy format
        using MPI I/O.

        Arrays written with this function can be read with numpy.load.

        Args:
            file_name (str): File name.
            vec (np.ndarray):
            Portion of the array on this MPI processes. This needs to be a
            two-dimensional array.
        """

        magic_str = magic(1, 0)
        x_size, y_size = vec.shape

        vx, vy = np.empty_like(x_size), np.empty_like(y_size)

        commx = self.rank_grid.Sub((True, False))
        commy = self.rank_grid.Sub((False, True))
        commx.Allreduce(np.asarray(x_size), vx)
        commy.Allreduce(np.asarray(y_size), vy)

        arr_dict_str = str({'descr': dtype_to_descr(vec.dtype),
                            'fortran_order': False,
                            'shape': (np.asscalar(vx), np.asscalar(vy))})

        while (len(arr_dict_str) + len(magic_str) + 2) % 16 != 15:
            arr_dict_str += ' '
        arr_dict_str += '\n'
        header_len = len(arr_dict_str) + len(magic_str) + 2

        x_offset = np.zeros_like(x_size)
        commx.Exscan(np.asarray(vy * x_size), x_offset)
        y_offset = np.zeros_like(y_size)
        commy.Exscan(np.asarray(y_size), y_offset)

        file = MPI.File.Open(self.rank_grid, file_name, MPI.MODE_CREATE | MPI.MODE_WRONLY)
        if self.rank == 0:
            file.Write(magic_str)
            file.Write(np.int16(len(arr_dict_str)))
            file.Write(arr_dict_str.encode('latin-1'))
        mpitype = MPI._typedict[vec.dtype.char]
        filetype = mpitype.Create_vector(x_size, y_size, vy)
        filetype.Commit()
        file.Set_view(header_len + (y_offset + x_offset) * mpitype.Get_size(),
                      filetype=filetype)
        file.Write_all(vec.copy())
        filetype.Free()
        file.Close()
