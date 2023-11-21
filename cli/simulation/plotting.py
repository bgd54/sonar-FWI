import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.animation as animation

mpl.rc("font", size=16)
mpl.rc("figure", figsize=(8, 6))


def _concat_matrix(vp_data, num_processes):
    if num_processes == 2:
        return np.concatenate(vp_data, axis=0)
    elif num_processes == 4:
        top_half = np.concatenate([vp_data[0], vp_data[1]], axis=0)
        bottom_half = np.concatenate([vp_data[2], vp_data[3]], axis=0)
        return np.concatenate([top_half, bottom_half], axis=1)
    else:
        raise ValueError("Number of processes not supported")


def _handle_mpi_communication(model, source=None, receiver=None, result_coords=None):
    nprocs = model.grid.distributor.nprocs
    if nprocs > 1:
        comm = model.grid.distributor.comm
        rank = model.grid.distributor.myrank
        vp_data = comm.gather(model.vp.data, root=0)
        if source is not None:
            source = comm.gather(source, root=0)
            if rank == 0:
                source = np.concatenate(source, axis=0)

        if receiver is not None:
            receiver = comm.gather(receiver, root=0)
            if rank == 0:
                receiver = np.concatenate(receiver, axis=0)

        if result_coords is not None:
            result_coords = comm.gather(result_coords, root=0)
            if rank == 0:
                result_coords = np.concatenate(result_coords, axis=0)

        if rank != 0:
            return None, None, None, None, False

        vp_data = _concat_matrix(vp_data, nprocs)
    else:
        vp_data = model.vp.data
    return vp_data, source, receiver, result_coords, True


def plot_velocity(
    model, source=None, receiver=None, result_coords=None, colorbar=True, cmap="jet"
):
    """
    Plot a two-dimensional velocity field from a seismic `Model`
    object. Optionally also includes point markers for sources and receivers.

    Parameters
    ----------
    model : Model
        Object that holds the velocity model.
    source : array_like or float
        Coordinates of the source point.
    receiver : array_like or float
        Coordinates of the receiver points.
    colorbar : bool
        Option to plot the colorbar.
    """

    (
        vp_data,
        source,
        receiver,
        result_coords,
        should_continue,
    ) = _handle_mpi_communication(model, source, receiver, result_coords)

    if not should_continue:
        # Terminate early for non-zero ranks in a parallel environment
        return

    domain_size = np.array(model.domain_size)
    extent = [
        model.origin[0],
        model.origin[0] + domain_size[0],
        model.origin[1] + domain_size[1],
        model.origin[1],
    ]

    slices = tuple(slice(model.nbl, -model.nbl) for _ in range(2))
    field = vp_data[slices]
    plot = plt.imshow(
        np.transpose(field),
        animated=True,
        cmap=cmap,
        vmin=np.min(field),
        vmax=np.max(field),
        extent=extent,
    )
    plt.xlabel("X position (m)")
    plt.ylabel("Depth (m)")
    if result_coords is not None:
        plt.title("Velocity profile + sources & receivers")
    else:
        plt.title("Velocity profile + sources & receivers + result")
    plt.xlim(model.origin[0], model.origin[0] + domain_size[0])
    plt.ylim(model.origin[1] + domain_size[1], model.origin[1])

    if receiver is not None:
        plt.scatter(receiver[:, 0], receiver[:, 1], s=25, c="green", marker="D")

    if source is not None:
        plt.scatter(source[:, 0], source[:, 1], s=25, c="red", marker="o")

    if result_coords is not None:
        plt.scatter(
            result_coords[:, 0], result_coords[:, 1], s=25, c="black", marker="x"
        )

    if colorbar:
        ax = plt.gca()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(plot, cax=cax)
        cbar.set_label("Velocity (km/s)")
    plt.show()


# Utility function to generate an animation from a snapshot, given a simulation - highlighting source and receiver positions
def plot_snapshot_and_signal(
    snap: npt.NDArray,
    recording: npt.NDArray,
    model,
    outfile,
    source_coords=None,
    receiver_coords=None,
):
    fig, ax = plt.subplots(
        2,
        1,
        figsize=(10, 15),
        gridspec_kw={"width_ratios": [1], "height_ratios": [2, 1]},
    )
    snap_step = int(recording.shape[0] / snap.shape[0])
    domain_size = np.array(model.domain_size)
    extent = [
        model.origin[0],
        model.origin[0] + domain_size[0],
        model.origin[1] + domain_size[1],
        model.origin[1],
    ]
    slices = tuple(slice(model.nbl, -model.nbl) for _ in range(2))
    field = model.vp.data[slices]
    ax[0].imshow(
        np.transpose(field),
        cmap="viridis",
        vmin=np.min(field),
        vmax=np.max(field),
        extent=extent,
    )

    if receiver_coords is not None:
        ax[0].scatter(
            receiver_coords[:, 0], receiver_coords[:, 1], s=25, c="green", marker="o"
        )
    if source_coords is not None:
        ax[0].scatter(
            source_coords[:, 0], source_coords[:, 1], s=25, c="red", marker="o"
        )

    ampl_limit = max(abs(np.min(snap)), abs(np.max(snap))) / 2
    snap = snap[:, model.nbl : -model.nbl, model.nbl : -model.nbl]
    matrice = ax[0].imshow(
        snap[0, :, :].T,
        vmin=-ampl_limit,
        vmax=ampl_limit,
        cmap="seismic",
        alpha=0.6,
        extent=extent,
    )

    (aline,) = ax[1].plot(recording[:1])
    ax[1].set_ylim(1.1 * np.min(recording), 1.1 * np.max(recording))
    ax[1].set_xlim(0, len(recording))

    def update(i):
        matrice.set_array(snap[i, :, :].T)
        aline.set_data(np.arange(i * snap_step), recording[: i * snap_step])
        return (matrice, aline)

    ani = animation.FuncAnimation(
        fig, update, frames=snap.shape[0], interval=75, blit=True
    )
    ani.save(outfile)
    plt.close(ani._fig)
