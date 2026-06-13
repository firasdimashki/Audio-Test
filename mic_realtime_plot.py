import argparse
from collections import deque
import threading

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sounddevice as sd


def dc_block_filter(
    samples: np.ndarray,
    *,
    alpha: float,
    previous_input: float,
    previous_output: float,
) -> tuple[np.ndarray, float, float]:
    """First-order high-pass filter, like an AC-coupling capacitor."""
    filtered = np.empty_like(samples, dtype=np.float32)
    x_prev = previous_input
    y_prev = previous_output

    for idx, x in enumerate(samples.astype(np.float32, copy=False)):
        y = alpha * (y_prev + float(x) - x_prev)
        filtered[idx] = y
        x_prev = float(x)
        y_prev = float(y)

    return filtered, x_prev, y_prev


def main() -> None:
    parser = argparse.ArgumentParser(description="Realtime microphone waveform plot.")
    parser.add_argument("--samplerate", type=int, default=44100, help="Sample rate (Hz).")
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
        "--dc-block-cutoff",
        type=float,
        default=20.0,
        help=(
            "High-pass cutoff in Hz for simulating a series coupling capacitor "
            "that removes DC offset. Use 0 to disable."
        ),
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
    dc_block_cutoff = float(args.dc_block_cutoff)
    if dc_block_cutoff < 0:
        raise SystemExit("--dc-block-cutoff must be >= 0.")

    if dc_block_cutoff > 0:
        rc = 1.0 / (2.0 * np.pi * dc_block_cutoff)
        dt = 1.0 / float(samplerate)
        dc_block_alpha = rc / (rc + dt)
    else:
        dc_block_alpha = 0.0

    # Ring buffer of recent samples filled by the audio callback.
    audio_buffer = deque(maxlen=window_samples)
    buffer_lock = threading.Lock()
    last_status: str = ""
    filter_previous_input = 0.0
    filter_previous_output = 0.0

    def audio_callback(indata: np.ndarray, frames: int, time, status) -> None:
        nonlocal filter_previous_input, filter_previous_output, last_status
        if status:
            # Keep the message; update the UI on the main thread.
            last_status = str(status)

        # indata is typically shaped (frames, channels). Downmix to mono for plotting.
        if indata.ndim == 2 and indata.shape[1] > 1:
            mono = np.mean(indata, axis=1)
        else:
            mono = np.asarray(indata).reshape(-1)

        if dc_block_cutoff > 0:
            mono, filter_previous_input, filter_previous_output = dc_block_filter(
                mono,
                alpha=dc_block_alpha,
                previous_input=filter_previous_input,
                previous_output=filter_previous_output,
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

    if dc_block_cutoff > 0:
        ax.set_title(f"Microphone waveform (DC blocked, {dc_block_cutoff:g} Hz cutoff)")
    else:
        ax.set_title("Microphone waveform (realtime)")
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

