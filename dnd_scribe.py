import whisper
import ollama
import torch
import os
import time
import argparse
import sys

# =====================================================================
# CONFIGURATION
# =====================================================================
# Whisper models: 'tiny', 'base', 'small', 'medium', 'large-v3'
WHISPER_MODEL_NAME = "medium"
# =====================================================================

def get_ollama_model():
    """Queries the local Ollama instance and lets the user pick a model."""
    print("Fetching available Ollama models...")
    try:
        response = ollama.list()
        
        # Handle different versions of the ollama python package (dict vs object)
        models_list = getattr(response, 'models', None)
        if models_list is None and isinstance(response, dict):
            models_list = response.get('models', [])
            
        if not models_list:
            print("No models found in Ollama. Please pull a model first (e.g., 'ollama pull llama3.1').")
            sys.exit(1)

        print("\nInstalled Ollama Models:")
        model_names = []
        for i, m in enumerate(models_list):
            # Extract the string name safely regardless of library version
            if isinstance(m, dict):
                name = m.get('model', m.get('name', 'unknown'))
            else:
                name = getattr(m, 'model', getattr(m, 'name', 'unknown'))
            
            model_names.append(name)
            print(f"[{i + 1}] {name}")

        # Prompt the user to select one
        while True:
            try:
                choice = int(input("\nEnter the number of the model you want to use: "))
                if 1 <= choice <= len(model_names):
                    selected = model_names[choice - 1]
                    print(f"Selected model: {selected}\n")
                    return selected
                print("Invalid number. Please try again.")
            except ValueError:
                print("Please enter a valid number.")

    except Exception as e:
        print(f"Error communicating with Ollama: {e}")
        print("Make sure the Ollama app is running in the background.")
        sys.exit(1)

def transcribe_recording(file_path):
    print(f"Loading Whisper model '{WHISPER_MODEL_NAME}'...")
    try:
        model = whisper.load_model(WHISPER_MODEL_NAME)
    except Exception as e:
        print(f"Error loading Whisper model: {e}")
        return None

    print(f"Transcribing '{file_path}'... (This might take a while depending on file length)")
    start_time = time.time()
    
    result = model.transcribe(file_path, fp16=torch.cuda.is_available(), condition_on_previous_text=False)
    
    end_time = time.time()
    print(f"Transcription complete in {round(end_time - start_time, 2)} seconds.")
    
    return result["text"]

def summarize_with_ollama(transcript, model_name):
    print(f"Sending transcript to Ollama model '{model_name}'...")
    
    system_prompt = (
        "You are an expert tabletop RPG archivist and storyteller. Read the following raw session transcript. "
        "Your task is to separate the out-of-character (OOC) player banter, jokes, and rule discussions from "
        "the actual in-character (IC) narrative.\n\n"
        "Please provide a comprehensive summary of the session formatted in Markdown. Include the following sections:\n"
        "1. **Session Overview**: A brief summary of what happened.\n"
        "2. **Locations Visited**: Where the party went.\n"
        "3. **NPCs Met/Interacted With**: Who they spoke to and key takeaways.\n"
        "4. **Key Plot Points & Lore**: The most important story beats and revelations.\n"
        "5. **Loot & Items**: Any notable items acquired or lost.\n"
        "6. **Combat Outcomes**: Brief summary of any battles.\n\n"
        "Completely ignore discussions about ordering food, scheduling, dice math, and off-topic jokes."
    )

    try:
        response = ollama.chat(model=model_name, messages=[
            {
                'role': 'system',
                'content': system_prompt
            },
            {
                'role': 'user',
                'content': f"Here is the raw session transcript to summarize:\n\n{transcript}"
            }
        ])
        return response['message']['content']
    except Exception as e:
        print(f"Error communicating with Ollama: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Transcribe and summarize D&D session recordings locally.")
    parser.add_argument("input_file", help="The name or path of the recording file (e.g., session1.mkv)")
    parser.add_argument("-m", "--model", help="Skip the menu and specify the Ollama model to use", default=None)
    args = parser.parse_args()

    input_file = args.input_file

    if not os.path.exists(input_file):
        print(f"Error: Could not find the file '{input_file}'.")
        return

    # 1. Determine which model to use
    if args.model:
        ollama_model_name = args.model
        print(f"Using pre-selected model: {ollama_model_name}\n")
    else:
        ollama_model_name = get_ollama_model()

    # Create dynamic output names based on the input file
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    raw_transcript_file = f"{base_name}_raw_transcript.txt"
    summary_file = f"{base_name}_summary.md"

    # 2. Transcribe the audio/video
    transcript = transcribe_recording(input_file)
    if not transcript:
        return

    with open(raw_transcript_file, "w", encoding="utf-8") as f:
        f.write(transcript)
    print(f"Saved a backup of the raw transcript to '{raw_transcript_file}'.")

    # 3. Summarize the text using the selected Ollama model
    summary = summarize_with_ollama(transcript, ollama_model_name)
    if not summary:
        return

    # 4. Save the final summary
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"\nSuccess! Your organized session notes have been saved to '{summary_file}'.")

if __name__ == "__main__":
    main()
