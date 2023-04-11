from filters import fir_filter, iir_filter, fourier_transform
from numpy import cos, sin, pi, absolute, arange
import numpy as np
from matplotlib import pyplot as plt

if __name__ == "__main__":
    # Create signal 
    np.random.seed(42)  # for reproducibility
    # create time steps and corresponding sine wave with Gaussian noise
    fs = 80  # sampling rate, Hz
    ts = np.arange(0, 5, 1.0 / fs)  # time vector - 5 seconds
    x_t = np.sin(2*np.pi * 10.0 * ts)  # signal @ 1.0 Hz, without noise
    noise = 0.2*sin(2*pi*15.3*ts) + 0.1*sin(2*pi*16.7*ts + 0.1) + 0.1*sin(2*pi*23.45*ts+.8)
    #noise = 0.5 * np.random.normal(size=len(ts))  # Gaussian noise
    x_noise = x_t + noise

    # plot raw signal
    plt.figure(figsize=[6.4, 2.4])
    plt.plot(ts, x_t, alpha=0.8, lw=3, color="C1", label="Clean signal (ys)")
    plt.plot(ts, x_noise, color="C0", label="Noisy signal (x_noise)")
    plt.xlabel("Time / s")
    plt.ylabel("Amplitude")
    plt.legend(loc="lower center", bbox_to_anchor=[0.5, 1], ncol=2,
            fontsize="smaller")
    plt.tight_layout()
    plt.show()


    # Signal Fourier Transform before filtering
    xf, yf = fourier_transform(x_noise, sample_rate=fs, duration=5)
    plt.plot(xf, np.abs(yf))
    plt.show()

    # define lowpass filter with 2.5 Hz cutoff frequency
    fc = 15
    x_filtered = iir_filter(x_noise, fc, fs)

    # Signal Fourier Transform after filtering
    xf, yf = fourier_transform(x_filtered, sample_rate=fs, duration=5)
    plt.plot(xf, np.abs(yf))
    plt.show()

    plt.figure(figsize=[6.4, 2.4])
    plt.plot(ts, x_noise, label="Raw signal")
    plt.plot(ts, x_filtered, alpha=0.8, lw=3, label="SciPy lfilter")
    plt.xlabel("Time / s")
    plt.ylabel("Amplitude")
    plt.legend(loc="lower center", bbox_to_anchor=[0.5, 1], ncol=2,
            fontsize="smaller")
    plt.tight_layout()

    # apply filter forward and backward using filtfilt
    x_filtered_fbf = iir_filter(x_noise, fc, fs, fbf=True)
    plt.figure(figsize=[6.4, 2.4])
    plt.plot(ts, x_noise, label="Raw signal")
    plt.plot(ts, x_filtered, alpha=0.5, lw=3, label="SciPy lfilter")
    plt.plot(ts, x_filtered_fbf, alpha=0.8, lw=3, label="SciPy filtfilt")
    plt.legend(loc="lower center", bbox_to_anchor=[0.5, 1], ncol=3,
            fontsize="smaller")
    plt.xlabel("Time / s")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()

    
