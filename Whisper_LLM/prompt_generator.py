import whisper
import os
import sys

def transcribe_audio(file_path):
    """Uses Whisper to transcribe the audio file and split it into sentences."""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return []

    print("Loading Whisper model...")
    model = whisper.load_model("large")
    print("Transcribing audio...")
    result = model.transcribe(file_path)

    # Collect segments into a list for further processing
    segments = [
        {
            "start": segment["start"],
            "end": segment["end"],
            "text": segment["text"]
        }
        for segment in result["segments"]
    ]

    return segments

def generate_prompt_string(segments, audio_name):
    """Generate a prompt for labeling speakers in a 911 call."""
    prompt = (
        "You are analyzing a 911 emergency audio call. Identify the different speakers "
        "in the conversation below. If possible, label each segment with the speaker, "
        "using '911 Operator', 'Caller', or 'Narrator'. Here are the segments:\n\n"
    )

    for segment in segments:
        prompt += f"[{segment['start']:.2f}s - {segment['end']:.2f}s]: {segment['text']}\n"

    prompt += "\nLabel each segment with the corresponding speaker. If unsure, mark it as 'Unknown'.\n"
    prompt += f"\nGive the output in a csv file named {audio_name}_whisper_llm.csv with this column format: [start_time], [end_time], [speaker], [phrase].\n"
    return prompt

def save_prompt_to_file(audio_name, prompt):
    """Save the generated prompt to a file in the 'prompts' folder."""
    os.makedirs("prompts", exist_ok=True)

    prompt_file_path = os.path.join("prompts", f"{audio_name}.txt")

    with open(prompt_file_path, "w") as f:
        f.write(prompt)

    print(f"Prompt saved to: {prompt_file_path}")

def main(audio_path):
    # Step 1: Transcribe the audio and get segments
    segments = transcribe_audio(audio_path)
    if not segments:
        print("No segments found, exiting.")
        return

    audio_name = os.path.splitext(os.path.basename(audio_path))[0]
    # Step 2: Generate the prompt string
    prompt = generate_prompt_string(segments, audio_name)

    # Step 3: Save the prompt to a file
    save_prompt_to_file(audio_name, prompt)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <path_to_audio_file>")
        sys.exit(1)

    audio_file_path = sys.argv[1]
    main(audio_file_path)
