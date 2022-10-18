import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.signal import find_peaks, peak_prominences

mpl.rc('font', size=16)
mpl.rc('figure', figsize=(8, 6))


def compare_velocity_to_measure(model,
                                result_coords,
                                source=None,
                                receiver=None,
                                colorbar=True,
                                cmap="jet"):
    domain_size = np.array(model.domain_size)
    extent = [
        model.origin[0], model.origin[0] + domain_size[0],
        model.origin[1] + domain_size[1], model.origin[1]
    ]

    slices = tuple(slice(model.nbl, -model.nbl) for _ in range(2))
    if getattr(model, 'vp', None) is not None:
        field = model.vp.data[slices]
    else:
        field = model.lam.data[slices]
    plot = plt.imshow(np.transpose(field),
                      animated=True,
                      cmap=cmap,
                      vmin=np.min(field),
                      vmax=np.max(field),
                      extent=extent)
    plt.xlabel('X position (m)')
    plt.ylabel('Depth (km)')

    # Plot source points, if provided
    if receiver is not None:
        plt.scatter(receiver[:, 0],
                    receiver[:, 1],
                    s=25,
                    c='green',
                    marker='D')

    # Plot receiver points, if provided
    if source is not None:
        plt.scatter(source[:, 0], source[:, 1], s=25, c='red', marker='o')
    plt.scatter(result_coords[:, 0],
                result_coords[:, 1],
                s=25,
                c='red',
                marker='o')

    # Ensure axis limits
    plt.xlim(model.origin[0] - domain_size[0] * 0.1,
             model.origin[0] + domain_size[0] * 1.1)
    plt.ylim(model.origin[1] + domain_size[1] * 1.1,
             model.origin[1] - domain_size[1] * 0.1)

    # Create aligned colorbar on the right
    if colorbar:
        ax = plt.gca()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(plot, cax=cax)
        cbar.set_label('Velocity (km/s)')
    plt.show()


def plot_shotrecord2(recording, cutoff=600):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    X = np.arange(recording.shape[1])
    Y = np.arange(recording.shape[0] - cutoff)
    X, Y = np.meshgrid(X, Y)

    # Plot the surface.
    surf = ax.plot_surface(X,
                           Y,
                           recording[cutoff:],
                           cmap=cm.coolwarm,
                           linewidth=0,
                           antialiased=False)

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()


def peaks(receiver,
          timestep: float,
          v_env: float,
          cut: int = 600) -> tuple[np.typing.NDArray, np.typing.NDArray]:
    x = receiver[cut:]
    peaks, _ = find_peaks(x)
    prominences = peak_prominences(x, peaks)[0]
    return peaks + 600, prominences


def dist_to_peak(peak: int, timestep: float, v_env: float) -> float:
    return (peak * timestep) / 2000 * v_env


def first_peak(peaks: tuple[np.typing.NDArray, np.typing.NDArray]) -> int:
    first_peak = peaks[0][(peaks[1] -
                           np.average(peaks[1])) > np.std(peaks[1])][0]
    return first_peak


def plot_signal(sig, timestep, v, ax, cut=600):
    x = sig[cut:]
    peaks, _ = find_peaks(x)
    prominences = peak_prominences(x, peaks)[0]
    first_peak = cut + peaks[
        (prominences - np.average(prominences)) > np.std(prominences)]
    ax.plot(sig)
    ax.plot(cut + peaks, sig[cut + peaks], 'ro')
    ax.plot(first_peak, sig[first_peak], 'bx')
