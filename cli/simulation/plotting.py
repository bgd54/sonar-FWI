import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from enum import Enum
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.signal import find_peaks, peak_prominences
import matplotlib.animation as animation
from simulation import utils

mpl.rc("font", size=16)
mpl.rc("figure", figsize=(8, 6))


class PlotType(str, Enum):
    model = "model"
    compare_velocity_to_measure = "compare_velocity_to_measure"
    shotrecord = "shotrecord"
    signal = "signal"


def compare_velocity_to_measure(
    model, result_coords, source=None, receiver=None, colorbar=True, cmap="jet"
):
    """
    Plot the velocity of the water and the measured distance of the object from the receiver.

    Args:
        model (Model): Model of the simulation.
        result_coords (np.ndarray): Coordinates of the result.
        source (Source, optional): Source of the signal. Defaults to None.
        receiver (Receiver, optional): Receiver of the signal. Defaults to None.
        colorbar (bool, optional): Show colorbar. Defaults to True.
        cmap (str, optional): Colormap. Defaults to "jet".
    """
    domain_size = np.array(model.domain_size)
    extent = [
        model.origin[0],
        model.origin[0] + domain_size[0],
        model.origin[1] + domain_size[1],
        model.origin[1],
    ]

    slices = tuple(slice(model.nbl, -model.nbl) for _ in range(2))
    if getattr(model, "vp", None) is not None:
        field = model.vp.data[slices]
    else:
        field = model.lam.data[slices]
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

    # Plot source points, if provided
    if receiver is not None:
        plt.scatter(receiver[:, 0], receiver[:, 1], s=25, c="green", marker="D")

    # Plot receiver points, if provided
    if source is not None:
        plt.scatter(source[:, 0], source[:, 1], s=25, c="red", marker="o")
    plt.scatter(result_coords[:, 0], result_coords[:, 1], s=25, c="black", marker="x")

    # Ensure axis limits
    plt.xlim(
        model.origin[0] - domain_size[0] * 0.1, model.origin[0] + domain_size[0] * 1.1
    )
    plt.ylim(
        model.origin[1] + domain_size[1] * 1.1, model.origin[1] - domain_size[1] * 0.1
    )

    # Create aligned colorbar on the right
    if colorbar:
        ax = plt.gca()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(plot, cax=cax)
        cbar.set_label("Velocity (km/s)")


def plot_shotrecord2(recording, cutoff=600):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    X = np.arange(recording.shape[1])
    Y = np.arange(recording.shape[0] - cutoff)
    X, Y = np.meshgrid(X, Y)

    # Plot the surface.
    surf = ax.plot_surface(
        X, Y, recording[cutoff:], cmap=mpl.cm.coolwarm, linewidth=0, antialiased=False
    )

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()


def peaks(
    receiver, timestep: float, v_env: float, cut: int = 600
) -> tuple[np.typing.NDArray, np.typing.NDArray]:
    x = receiver[cut:]
    peaks, _ = find_peaks(x)
    prominences = peak_prominences(x, peaks)[0]
    return peaks + 600, prominences


def dist_to_peak(peak: int, timestep: float, v_env: float) -> float:
    return (peak * timestep) / 2000 * v_env


def first_peak(peaks: tuple[np.typing.NDArray, np.typing.NDArray]) -> int:
    first_peak = peaks[0][(peaks[1] - np.average(peaks[1])) > np.std(peaks[1])][0]
    return first_peak


def plot_signal(sig, timestep, v, ax, cut=600):
    x = sig[cut:]
    peaks, _ = find_peaks(x)
    prominences = peak_prominences(x, peaks)[0]
    first_peak = (
        cut + peaks[(prominences - np.average(prominences)) > np.std(prominences)]
    )
    ax.plot(sig)
    ax.plot(cut + peaks, sig[cut + peaks], "ro")
    ax.plot(first_peak, sig[first_peak], "bx")


def plot_velocity(model, source=None, receiver=None, colorbar=True, cmap="jet"):
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
    domain_size = 1.0e-3 * np.array(model.domain_size)
    extent = [
        model.origin[0],
        model.origin[0] + domain_size[0],
        model.origin[1] + domain_size[1],
        model.origin[1],
    ]

    slices = tuple(slice(model.nbl, -model.nbl) for _ in range(2))
    if getattr(model, "vp", None) is not None:
        field = model.vp.data[slices]
    else:
        field = model.lam.data[slices]
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

    # Plot source points, if provided
    if receiver is not None:
        plt.scatter(
            1e-3 * receiver[:, 0], 1e-3 * receiver[:, 1], s=25, c="green", marker="D"
        )

    # Plot receiver points, if provided
    if source is not None:
        plt.scatter(1e-3 * source[:, 0], 1e-3 * source[:, 1], s=25, c="red", marker="o")

    # Ensure axis limits
    plt.xlim(model.origin[0], model.origin[0] + domain_size[0])
    plt.ylim(model.origin[1] + domain_size[1], model.origin[1])

    # Create aligned colorbar on the right
    if colorbar:
        ax = plt.gca()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(plot, cax=cax)
        cbar.set_label("Velocity (km/s)")
    plt.show()


def plot_snapshot_and_signal(snap: npt.NDArray, recording: npt.NDArray, model, outfile):
    dt = model.critical_dt
    v_env = model.vp.data[int(model.vp.data.shape[0] / 2), 0]
    snap_step = int(recording.shape[0] / snap.shape[0])
    aline_data = np.average(recording.data, axis=1)
    first_peak = utils.first_peak_after(aline_data, dt, v_env, cut_ms=10.0)

    fig, axs = plt.subplots(
        2, 1, gridspec_kw={"width_ratios": [1], "height_ratios": [2, 1]}
    )
    extent = [
        model.origin[0],
        model.origin[0] + model.domain_size[0],
        model.origin[1] + model.domain_size[1],
        model.origin[1],
    ]
    axs[0].imshow(
        np.transpose(model.vp.data),
        cmap="viridis",
        vmin=np.min(model.vp.data),
        vmax=np.max(model.vp.data),
        extent=extent,
    )

    ampl_limit = max(abs(np.min(snap)), abs(np.max(snap)))

    matrice = axs[0].imshow(
        snap[0, :, :].T,
        vmin=-ampl_limit,
        vmax=ampl_limit,
        alpha=0.6,
        extent=extent,
        cmap="seismic",
    )
    fig.colorbar(matrice, ax=axs[0])
    (aline,) = axs[1].plot(aline_data[:1])
    (detection,) = axs[1].plot([], [], "rx")
    # label = axs[1].text(2000, 0.0004, f"t=0")

    axs[0].set(xlabel="X position (m)", ylabel="Depth (m)")
    axs[1].set_xlim(0, aline_data.shape[0])
    axs[1].set_ylim(
        1.1 * np.min(np.average(recording.data, axis=1)),
        1.1 * np.max(np.average(recording.data, axis=1)),
    )
    ticks = utils.num_iter_for_distance(
        np.arange(
            0, round(utils.object_distance_iter(aline_data.shape[0], dt, v_env)), 2
        ),
        dt,
        v_env,
    )
    par1 = axs[1].twiny()
    par1.set_xlabel("Distance (m)")
    par1.set_xticks(ticks)

    par1.set_xticklabels(np.arange(len(ticks)) * 2)
    fig.tight_layout()

    def update(i):
        matrice.set_array(snap[i, :, :].T)
        aline.set_data(np.arange(i * snap_step), aline_data[: i * snap_step])
        if i * snap_step > first_peak:
            detection.set_data(first_peak, aline_data[first_peak])
        else:
            detection.set_data([], [])

        # label.set_text(f"t={i*snap_step*dt:.4f} ms")
        return (
            matrice,
            aline,
            detection,
            # label,
        )

    # Animation
    ani = animation.FuncAnimation(
        fig, func=update, frames=snap.shape[0], interval=75, blit=True
    )
    ani.save(outfile)
    plt.show()
