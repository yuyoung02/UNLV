import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np

# --- Configuration ---
DURATION = 2  # Duration in seconds
SAMPLE_RATE = 22050  # Standard for ML (use 44100 for high quality audio)
OUTPUT_FILENAME = "rpi_mic_recording.wav"


def list_input_devices():
    """Lists all available input devices to find the USB Mic."""
    print("\n--- Available Audio Devices ---")
    devices = sd.query_devices()
    input_devices = []

    for i, device in enumerate(devices):
        # We only care about devices that have input channels (microphones)
        if device['max_input_channels'] > 0:
            print(f"ID {i}: {device['name']}")
            input_devices.append(i)
    return input_devices


def record_from_device(device_id):
    print(f"\nPreparing to record for {DURATION} seconds using Device ID {device_id}...")

    try:
        # Start recording
        # dtype='int16' is standard for WAV files
        audio_data = sd.rec(int(DURATION * SAMPLE_RATE),
                            samplerate=SAMPLE_RATE,
                            channels=1,
                            device=device_id,
                            dtype='int16')

        print("Recording... (Speak now!)")
        sd.wait()  # Wait until the recording is finished
        print("Finished.")

        # Save to file
        write(OUTPUT_FILENAME, SAMPLE_RATE, audio_data)
        print(f"Saved successfully to '{OUTPUT_FILENAME}'")

    except Exception as e:
        print(f"Error recording: {e}")


if __name__ == "__main__":
    # 1. List devices so user can identify the USB Mic
    valid_ids = list_input_devices()

    # 2. Ask user to select the specific ID
    print("\nCheck the list above. Find your 'USB Audio Device' or similar name.")
    try:
        selected_id = int(input("Enter the ID number of your USB Microphone: "))

        if selected_id in valid_ids:
            record_from_device(selected_id)
        else:
            # Fallback check in case the ID is valid but not in our filtered list
            # (sometimes sd lists inputs weirdly on different OSs)
            record_from_device(selected_id)

    except ValueError:
        print("Invalid input. Please enter a number.")