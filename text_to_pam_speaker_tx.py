import argparse
from typing import List

import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd


def int_to_bits(value: int, bit_count: int) -> List[int]:
    """Return MSB-first bit list of fixed width."""
    if value < 0:
        raise ValueError("value must be non-negative")
    bits = [(value >> i) & 1 for i in range(bit_count - 1, -1, -1)]
    return bits


def bytes_to_bits(data: bytes, bits_per_byte: int = 8) -> List[int]:
    if bits_per_byte != 8:
        raise ValueError("Only bits_per_byte=8 is supported for now.")
    out: List[int] = []
    for b in data:
        out.extend(int_to_bits(b, 8))
    return out


def pam2_encode_nrz(bits: List[int], *, amplitude: float, samples_per_symbol: int) -> np.ndarray:
    """PAM2 (two-level) NRZ: bit=0 -> -amplitude, bit=1 -> +amplitude."""
    if samples_per_symbol <= 0:
        raise ValueError("samples_per_symbol must be > 0")
    symbols = np.where(np.asarray(bits) == 1, amplitude, -amplitude).astype(np.float32)
    # Repeat each symbol for its symbol duration.
    return np.repeat(symbols, samples_per_symbol)


def multiply_with_sine_carrier(
    baseband: np.ndarray,
    *,
    carrier_hz: float,
    samplerate: int,
) -> np.ndarray:
    """Modulate the PAM baseband by multiplying it with a sine carrier."""
    sample_idx = np.arange(baseband.size, dtype=np.float64)
    phase = 2.0 * np.pi * carrier_hz * sample_idx / float(samplerate)
    carrier = np.sin(phase).astype(np.float32)
    return (baseband.astype(np.float32, copy=False) * carrier).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Transmit terminal text over your speakers using PAM2 multiplied by a sine carrier. "
            "Encodes as: [length (big-endian, 16-bit)] + [UTF-8 bytes]."
        )
    )
    parser.add_argument("--baud", type=float, default=500.0, help="Symbol rate in baud.")
    parser.add_argument("--samplerate", type=int, default=48000, help="Audio sample rate (Hz).")
    parser.add_argument("--channels", type=int, default=1, help="Output channels (1=mono, 2=stereo).")
    parser.add_argument(
        "--encoding",
        type=str,
        default="utf-8",
        help="Text encoding used to convert the string into bytes.",
    )
    parser.add_argument(
        "--length-bits",
        type=int,
        default=16,
        help="Number of bits used for the byte-length header (big-endian).",
    )
    parser.add_argument(
        "--amplitude",
        type=float,
        default=0.1,
        help=(
            "Peak amplitude for PAM levels (0.0-1.0). "
            "Keep this low to avoid loud output."
        ),
    )
    parser.add_argument(
        "--carrier-hz",
        type=float,
        default=5000.0,
        help="Sine carrier frequency in Hz multiplied with the PAM signal.",
    )
    parser.add_argument(
        "--gap-ms",
        type=float,
        default=1000.0,
        help="Silence (ms) added before the frame.",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Optional text to send. If omitted, reads one line from stdin.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Do not show a waveform plot before transmission.",
    )
    parser.add_argument(
        "--plot-max-points",
        type=int,
        default=200_000,
        help="Max points to draw in the waveform plot (downsamples if needed).",
    )

    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List audio output devices supported by sounddevice and exit.",
    )
    args = parser.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    if not (0.0 < args.amplitude <= 1.0):
        raise SystemExit("--amplitude must be in (0.0, 1.0].")

    if args.channels not in (1, 2):
        raise SystemExit("--channels must be 1 or 2.")

    if not (0.0 < args.carrier_hz < args.samplerate / 2.0):
        raise SystemExit("--carrier-hz must be > 0 and below the Nyquist frequency.")

    # We require an integer number of samples per symbol so that symbol boundaries are exact.
    ratio = args.samplerate / args.baud
    samples_per_symbol = int(round(ratio))
    if abs(ratio - samples_per_symbol) > 1e-6:
        raise SystemExit(
            f"samplerate/baud must be an integer for exact symbol timing. "
            f"Got samplerate={args.samplerate}, baud={args.baud} => {ratio}. "
            f"Try samplerate=48000 for 9600 baud."
        )

    if args.text is None:
        # Read a single line from the terminal.
        args.text = input("Enter text to transmit: ")

    payload = args.text.encode(args.encoding)
    length = len(payload)
    if length >= (1 << args.length_bits):
        raise SystemExit(
            f"Message too long: {length} bytes does not fit in {args.length_bits} length bits."
        )

    bits = []
    bits.extend(int_to_bits(length, args.length_bits))  # framing
    bits.extend(bytes_to_bits(payload))

    waveform = pam2_encode_nrz(
        bits,
        amplitude=float(args.amplitude),
        samples_per_symbol=samples_per_symbol,
    )

    # Add zero-amplitude guard intervals before the frame.
    gap_samples = int(round(args.samplerate * (args.gap_ms / 1000.0)))
    if gap_samples > 0:
        zeros = np.zeros(gap_samples, dtype=np.float32)
        waveform = np.concatenate([zeros, waveform])

    waveform = multiply_with_sine_carrier(
        waveform,
        carrier_hz=float(args.carrier_hz),
        samplerate=int(args.samplerate),
    )

    # Convert to shape (N, channels) for sounddevice.
    if args.channels == 2:
        waveform = np.column_stack([waveform, waveform]).astype(np.float32)
    else:
        waveform = waveform.astype(np.float32)

    def plot_waveform(wf: np.ndarray) -> None:
        # Plot mono for readability even when transmitting stereo (both channels identical).
        y = wf[:, 0] if wf.ndim == 2 else wf
        n = int(y.size)
        step = max(1, n // max(2, int(args.plot_max_points)))
        t_ms = (np.arange(0, n, step, dtype=np.float64) / float(args.samplerate)) * 1000.0
        y_plot = y[::step].astype(np.float64)

        plt.figure(figsize=(10, 4))
        plt.plot(t_ms, y_plot, linewidth=1.0)
        plt.title(f"PAM2 waveform on {args.carrier_hz:g} Hz sine carrier")
        plt.xlabel("Time (ms)")
        plt.ylabel("Amplitude")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    # Play once then wait for completion.
    # Note: This transmits continuously for the entire encoded frame duration.

    print(
        f"Transmitting {len(payload)} byte(s) at {args.baud} baud "
        f"({samples_per_symbol} samples/symbol) using PAM2 on a "
        f"{args.carrier_hz:g} Hz sine carrier."
    )
    if not args.no_plot:
        plot_waveform(waveform)
        input("Close the plot window (or review it) then press Enter to transmit...")

    sd.play(waveform, samplerate=args.samplerate)
    sd.wait()


if __name__ == "__main__":
    main()

