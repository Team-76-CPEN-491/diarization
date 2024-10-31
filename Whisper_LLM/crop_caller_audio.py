import csv
import os
import sys
from pydub import AudioSegment

def parse_time(time_input):
    """
    Converts time input to milliseconds. Accepts:
    - Time string in the format 'MM:SS.mmm' (e.g., '02:14.250').
    - Floating-point seconds as input.
    # """
    if isinstance(time_input, str) and len(time_input.split(':')) > 1:
        try:
            # Split the time string into minutes and seconds with milliseconds
            minutes, seconds = time_input.split(':')

            minutes = int(minutes)
            seconds = float(seconds)  # Handle seconds with milliseconds
            
            # Convert to total seconds
            total_seconds = minutes * 60 + seconds
        except (ValueError, IndexError) as e:
            raise ValueError(f"Invalid time format: {time_input}") from e
    else:
        #Assume input is already in seconds if not a string
        total_seconds = float(time_input)

    # Convert to milliseconds and return as an integer
    return int(total_seconds * 1000)

def extract_911_audio(csv_file, input_audio_file):
    """Extracts 'Caller' segments from the audio and saves them in caller_audios folder."""
    # Load the audio file
    audio = AudioSegment.from_file(input_audio_file)
    audio_end_time = len(audio)
    segments = []

    # Read the CSV and identify relevant 'Caller' audio segments
    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        current_start = None

        for row in reader:
            speaker = row['speaker']
            start_time = parse_time(row['start_time'])

            if speaker == "Caller":
                # Start a new segment if needed
                if current_start is None:
                    current_start = start_time
            else:
                # Save the segment when the speaker changes from 'Caller'
                if current_start is not None:
                    segments.append((current_start, start_time))
                    current_start = None

        # Add the final segment if the audio ends with the 'Caller'
        if current_start is not None:
            segments.append((current_start, audio_end_time))

    # Combine all 'Caller' audio segments
    output_audio = AudioSegment.empty()
    for start, end in segments:
        output_audio += audio[start:end]

    # Prepare the output path and filename
    os.makedirs("audios_caller", exist_ok=True)
    original_ext = os.path.splitext(os.path.basename(input_audio_file))[1]
    original_name = os.path.splitext(os.path.basename(csv_file))[0]
    output_audio_file = os.path.join("audios_caller", f"{original_name}_caller{original_ext}")

    # Export the result
    output_audio.export(output_audio_file, format=original_ext.lstrip('.'))
    print(f"Output audio saved to {output_audio_file}")

def main():
    if len(sys.argv) < 3:
        print("Usage: python crop_caller_audio.py <path_to_csv_file> <path_to_audio_file>")
        sys.exit(1)

    csv_file = sys.argv[1]
    input_audio_file = sys.argv[2]

    extract_911_audio(csv_file, input_audio_file)

if __name__ == "__main__":
    main()
