import argparse
from collections import deque
import threading

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sounddevice as sd


def design_bandpass_biquad(
    *,
    center_hz: float,
    bandwidth_hz: float,
    samplerate: int,
) -> tuple[float, float, float, float, float]:
    """Return normalized biquad coefficients for a second-order band-pass filter."""
    q = center_hz / bandwidth_hz
    omega = 2.0 * np.pi * center_hz / float(samplerate)
    alpha = np.sin(omega) / (2.0 * q)
    cos_omega = np.cos(omega)
    a0 = 1.0 + alpha

    return (
        alpha / a0,
        0.0,
        -alpha / a0,
        (-2.0 * cos_omega) / a0,
        (1.0 - alpha) / a0,
    )


def apply_biquad_filter(
    samples: np.ndarray,
    *,
    coefficients: tuple[float, float, float, float, float],
    state: tuple[float, float],
) -> tuple[np.ndarray, tuple[float, float]]:
    """Apply a biquad filter with transposed direct-form II state."""
    b0, b1, b2, a1, a2 = coefficients
    z1, z2 = state
    filtered = np.empty_like(samples, dtype=np.float32)

    for idx, x in enumerate(samples.astype(np.float32, copy=False)):
        y = b0 * float(x) + z1
        z1 = b1 * float(x) - a1 * y + z2
        z2 = b2 * float(x) - a2 * y
        filtered[idx] = y

    return filtered, (z1, z2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Realtime microphone waveform plot.")
    parser.add_argument("--samplerate", type=int, default=48000, help="Sample rate (Hz).")
    parser.add_argument("--blocksize", type=int, default=1024, help="Audio block size (frames).")
    parser.add_argument(
        "--channels",
        type=int,
        default=1,
        help="Number of channels to capture (will be downmixed to mono for plotting).",
    )
    parser.add_argument(
        "--window_ms",
        type=int,
        default=2000,
        help="How much recent audio to show on screen (milliseconds).",
    )
    parser.add_argument(
        "--interval_ms",
        type=int,
        default=50,
        help="How often to redraw the plot (milliseconds).",
    )
    parser.add_argument("--device", type=str, default=None, help="sounddevice device index or None.")
    parser.add_argument(
        "--bandpass-center-hz",
        type=float,
        default=5000.0,
        help="Band-pass center frequency in Hz for receiving the carrier signal.",
    )
    parser.add_argument(
        "--bandpass-bandwidth-hz",
        type=float,
        default=2000.0,
        help="Band-pass bandwidth in Hz around the center frequency.",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List audio devices supported by sounddevice and exit.",
    )
    args = parser.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    samplerate = int(args.samplerate)
    blocksize = int(args.blocksize)
    channels = int(args.channels)
    window_samples = max(64, int(args.window_ms / 1000.0 * samplerate))
    bandpass_center_hz = float(args.bandpass_center_hz)
    bandpass_bandwidth_hz = float(args.bandpass_bandwidth_hz)
    bandpass_low_hz = bandpass_center_hz - (bandpass_bandwidth_hz / 2.0)
    bandpass_high_hz = bandpass_center_hz + (bandpass_bandwidth_hz / 2.0)
    nyquist_hz = samplerate / 2.0
    if bandpass_center_hz <= 0:
        raise SystemExit("--bandpass-center-hz must be > 0.")
    if bandpass_bandwidth_hz <= 0:
        raise SystemExit("--bandpass-bandwidth-hz must be > 0.")
    if bandpass_low_hz <= 0 or bandpass_high_hz >= nyquist_hz:
        raise SystemExit(
            "--bandpass-bandwidth-hz must keep the filter passband above 0 Hz "
            "and below the Nyquist frequency."
        )

    bandpass_coefficients = design_bandpass_biquad(
        center_hz=bandpass_center_hz,
        bandwidth_hz=bandpass_bandwidth_hz,
        samplerate=samplerate,
    )

    # Ring buffer of recent samples filled by the audio callback.
    audio_buffer = deque(maxlen=window_samples)
    buffer_lock = threading.Lock()
    last_status: str = ""
    bandpass_state = (0.0, 0.0)

    def audio_callback(indata: np.ndarray, frames: int, time, status) -> None:
        nonlocal bandpass_state, last_status
        if status:
            # Keep the message; update the UI on the main thread.
            last_status = str(status)

        # indata is typically shaped (frames, channels). Downmix to mono for plotting.
        if indata.ndim == 2 and indata.shape[1] > 1:
            mono = np.mean(indata, axis=1)
        else:
            mono = np.asarray(indata).reshape(-1)

        mono, bandpass_state = apply_biquad_filter(
            mono,
            coefficients=bandpass_coefficients,
            state=bandpass_state,
        )

        with buffer_lock:
            # Extend in one go to reduce callback overhead.
            audio_buffer.extend(mono.astype(np.float32).tolist())

    device = None
    if args.device is not None:
        # Accept either an int-like string or pass through.
        try:
            device = int(args.device)
        except ValueError:
            device = args.device

    try:
        stream = sd.InputStream(
            samplerate=samplerate,
            blocksize=blocksize,
            channels=channels,
            device=device,
            callback=audio_callback,
            dtype="float32",
        )
    except Exception as e:
        raise SystemExit(f"Failed to open audio stream: {e}")

    # --- Plot setup ---
    fig, ax = plt.subplots()
    fig.canvas.manager.set_window_title("Realtime Microphone Input")

    x_ms = np.linspace(-args.window_ms, 0, window_samples, dtype=np.float32)
    line, = ax.plot(x_ms, np.zeros(window_samples, dtype=np.float32), lw=1.2)
    status_text = ax.text(0.01, 0.95, "", transform=ax.transAxes, va="top")

    ax.set_title(
        f"Microphone waveform ({bandpass_center_hz:g} Hz band-pass, "
        f"{bandpass_bandwidth_hz:g} Hz BW)"
    )
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude")
    ax.set_xlim(-args.window_ms, 0)
    ax.set_ylim(-1.1, 1.1)
    ax.grid(True, alpha=0.3)

    def update(_frame_idx: int):
        nonlocal last_status
        with buffer_lock:
            y = np.asarray(audio_buffer, dtype=np.float32)
            status = last_status
            last_status = ""

        if y.size == 0:
            status_text.set_text(status)
            return (line, status_text)

        if y.size < window_samples:
            y_plot = np.concatenate([np.zeros(window_samples - y.size, dtype=np.float32), y])
        else:
            y_plot = y[-window_samples:]
        
        y_plot = np.clip(y_plot * 100, -1, 1)

        line.set_ydata(y_plot)
        status_text.set_text(status)
        return (line, status_text)

    anim = FuncAnimation(fig, update, interval=args.interval_ms, blit=False)

    # Start capture and block on the UI loop.
    with stream:
        try:
            plt.show()
        except KeyboardInterrupt:
            pass

    anim.event_source.stop()


if __name__ == "__main__":
    main()

