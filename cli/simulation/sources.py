import numpy as np
from examples.seismic import WaveletSource


class SineSource(WaveletSource):
    """
    A source object that encapsulates everything necessary for injecting a
    sine source into the computational domain.

    Returns:
        The source term that will be injected into the computational domain.
    """

    def __init_finalize__(self, *args, **kwargs):
        super(SineSource, self).__init_finalize__(*args, **kwargs)

    @property
    def wavelet(self):
        assert self.f0 is not None
        t0 = self.t0 or 1 / self.f0
        a = self.a or 1
        r = 2 * np.pi * self.f0 * (self.time_values - t0)
        wave = a * np.sin(r) + a * np.sin(3 * (r + np.pi) / 4)
        wave[np.searchsorted(self.time_values, 4 * 2 / self.f0) :] = 0
        return wave

    @property
    def signal_packet(self):
        assert self.f0 is not None
        return self.wavelet[: np.searchsorted(self.time_values, 4 * 2 / self.f0)]


class MultiFrequencySource(WaveletSource):
    """
    A source object that encapsulates everything necessary for injecting multiple
    sine source into the computational domain.

    Returns:
        The source term that will be injected into the computational domain.
    """

    def __init_finalize__(self, *args, **kwargs):
        self.alpha = kwargs.get("alpha")
        if self.alpha is None:
            self.alpha = np.array([90.0])
        self.packet_l = kwargs.get("packet_l") or 128
        super(MultiFrequencySource, self).__init_finalize__(*args, **kwargs)

    # property returning all wavelets with different frequencies to combine with the
    # latency profile

    @property
    def wavelet(self):
        return np.sum(self.wavelets, axis=0)

    @property
    def wavelets(self):
        assert self.f0 is not None and self.alpha is not None
        assert self.f0.shape == self.alpha.shape
        t0 = self.t0 or np.min(1 / self.f0)
        a = self.a or 1
        t = self.time_values - t0
        r = 2 * np.pi * self.f0[:, np.newaxis] * t
        wave = a * np.sin(r) + a * np.sin(
            (self.packet_l - 1) * (r + np.pi) / self.packet_l
        )
        for i, f in enumerate(self.f0):
            wave[i, np.searchsorted(t, self.packet_l * 2 / f) :] = 0

        return wave

    @property
    def signal_packet(self):
        assert self.f0 is not None
        return self.wavelet[
            : np.searchsorted(self.time_values, self.packet_l * 2 / np.min(self.f0))
        ]

    def apply_latency_profile(self, c):
        assert self.f0 is not None and self.alpha is not None
        # compute the latency shifts for each angle - wavelet pair and apply them
        # to set the data for each source
        assert all(
            map(lambda x: x[1] == self.coordinates.data[0, 1], self.coordinates.data)
        )
        source_distances = self.coordinates.data[:, 0] - self.coordinates.data[0, 0]
        latencies = -np.cos(np.deg2rad(self.alpha[:, np.newaxis])) * (
            source_distances / c
        )
        latencies_tick = (latencies / self.time_range.step).astype(int)
        latencies_tick = latencies_tick - np.min(latencies_tick, axis=1)[:, np.newaxis]
        self.data.fill(0)
        wavelets = self.wavelets
        for i in range(self.data.shape[1]):
            for j in range(latencies_tick.shape[0]):
                self.data[:, i] += np.roll(wavelets[j, :], latencies_tick[j, i])
