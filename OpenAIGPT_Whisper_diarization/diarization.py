import csv
import os
import sys
from pydub import AudioSegment
import whisper
import openai
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def parse_time(time_input):
    if isinstance(time_input, str) and len(time_input.split(':')) > 1:
        try:
            minutes, seconds = time_input.split(':')
            minutes = int(minutes)
            seconds = float(seconds)
            total_seconds = minutes * 60 + seconds
        except (ValueError, IndexError) as e:
            raise ValueError(f"Invalid time format: {time_input}") from e
    else:
        total_seconds = float(time_input)
    return int(total_seconds * 1000)

def extract_911_caller_audio(csv_file, input_audio_file):
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

            if speaker.strip() == "Caller":
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
    print(segments)
    for start, end in segments:
        output_audio += audio[start:end]

    # Prepare the output path and filename
    os.makedirs("audios_caller", exist_ok=True)
    original_ext = os.path.splitext(os.path.basename(input_audio_file))[1]
    original_name = os.path.splitext(os.path.basename(input_audio_file))[0]
    output_audio_file = os.path.join("audios_caller", f"{original_name}_caller{original_ext}")

    # Export the result
    output_audio.export(output_audio_file, format=original_ext.lstrip('.'))
    print(f"Output audio saved to {output_audio_file}")

def transcribe_audio(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return []
    print("Loading Whisper model...")
    model = whisper.load_model("large")
    print("Transcribing audio...")
    result = model.transcribe(file_path)
    segments = [
        {
            "start": segment["start"],
            "end": segment["end"],
            "text": segment["text"]
        }
        for segment in result["segments"]
    ]
    return segments

def call_openai_api(prompt):
    """Call OpenAI GPT-4 API with the given prompt."""
    try:
        response = client.chat.completions.create(model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that classifies audio transcription segments."},
            {"role": "user", "content": prompt}
        ])
        return response.choices[0].message.content
    except openai.OpenAIError as e:
        print(f"OpenAI API Error: {e}")
        return None

def generate_prompt_string(segments):
    prompt = (
        "You are analyzing a 911 emergency audio call. Label each segment with the speaker ('911 Operator', 'Caller', or 'Narrator'). "
        "If unsure, mark it as 'Unknown'.\n\n"
    )
    for segment in segments:
        prompt += f"[{segment['start']:.2f}, {segment['end']:.2f}]: {segment['text']}\n"
    prompt += "\nProvide the labeled segments in the format: start_time, end_time, speaker, phrase."
    prompt += "\nThis output should be in CSV format, please do not add extra character."
    return prompt

def classify_segments_with_gpt(audio_name, segments):
    """Classify transcription segments using GPT-4."""
    prompt = generate_prompt_string(segments)
    print("Calling OpenAI GPT-4 API...")
    gpt_response = call_openai_api(prompt)

    if gpt_response:
        # Save the GPT response to a CSV file
        os.makedirs("classified_transcripts", exist_ok=True)
        output_file = os.path.join("classified_transcripts", f"{audio_name}_classified_transcript.csv")
        with open(output_file, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["start_time", "end_time", "speaker", "phrase"])
            for line in gpt_response.strip().split('\n'):
                parts = line.split(',', 3)
                if len(parts) == 4:
                    writer.writerow(parts)
        print(f"Classification saved to {output_file}")
        return output_file
    else:
        print("Failed to get a response from GPT-4.")
        return None

def main(audio_path):
    segments = transcribe_audio(audio_path)
    if not segments:
        print("No segments found, exiting.")
        return
    audio_name = os.path.splitext(os.path.basename(audio_path))[0]
    csv_file = classify_segments_with_gpt(audio_name, segments)
    if csv_file:
        extract_911_caller_audio(csv_file, audio_path)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python diarization.py <path_to_audio_file>")
        sys.exit(1)
    audio_file_path = sys.argv[1]
    main(audio_file_path)
