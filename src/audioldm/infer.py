import os
from inference import (
    text_to_audio, style_transfer, 
    build_model, save_wave, 
    get_time, round_up_duration,
    get_duration
)
from unidecode import unidecode
import argparse

def inference(
    model_name: str,
    text: str,
    save_dir: str | None = None,
    duration: float = 10.0,
    mode: str = "generation",
    file_path: str | None = None,
    transfer_strength: float = 0.5,
    random_seed: int = 42,
    guidance_scale: float = 2.5,
    batchsize: int = 1,
    n_candidate_gen_per_text: int = 3,
    ddim_steps: int = 200,
    prompt_as_filename: bool = False
) -> tuple[str, any]:
    """
    Run inference with the given duration for audio generation or style transfer.

    Args:
        model_name: Name of the model checkpoint to use.
        text: Text prompt for audio generation or style transfer.
        save_dir: Directory to save the output audio. Defaults to "./output".
        duration: Duration of the generated audio in seconds.
        mode: Operation mode ("generation" or "transfer").
        file_path: Path to the input audio file for transfer or guidance.
        transfer_strength: Strength of style transfer (0 to 1).
        random_seed: Seed for reproducibility.
        guidance_scale: Guidance scale for quality vs. diversity.
        batchsize: Number of samples to generate at once.
        n_candidate_gen_per_text: Number of candidates for quality control.
        ddim_steps: Number of sampling steps for DDIM.
        prompt_as_filename: If True, use the text prompt as the filename.

    Returns:
        Tuple of (save directory path, generated waveform).
    """
    # Initialize model
    audioldm = build_model(model_name=model_name)

    # Handle mode override for generation with audio guidance
    if mode == "generation" and file_path is not None:
        mode = "generation_audio_to_audio"
        if text:
            print("Warning: --file_path provided; ignoring --text.")
            text = ""

    # Validate inputs based on mode
    if mode == "generation" and not text:
        raise ValueError("Text prompt is required for 'generation' mode.")
    if mode == "transfer":
        if file_path is None or not os.path.exists(file_path):
            raise ValueError(f"The original audio file '{file_path}' for style transfer does not exist.")

    # Set up save directory
    save_dir = save_dir or "./output"
    save_dir = os.path.join(save_dir, mode)
    if file_path:
        # Safely extract filename without extension
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        save_dir = os.path.join(save_dir, base_name)
    os.makedirs(save_dir, exist_ok=True)

    # Generate or transfer audio
    if mode in ["generation", "generation_audio_to_audio"]:
        waveform = text_to_audio(
            audioldm,
            text,
            file_path,
            random_seed,
            duration=duration,
            guidance_scale=guidance_scale,
            ddim_steps=ddim_steps,
            n_candidate_gen_per_text=n_candidate_gen_per_text,
            batchsize=batchsize,
        )
    elif mode == "transfer":
        waveform = style_transfer(
            audioldm,
            text,
            file_path,
            transfer_strength,
            random_seed,
            duration=duration,
            guidance_scale=guidance_scale,
            ddim_steps=ddim_steps,
            batchsize=batchsize,
        )
        # Note: Waveform reshaping (waveform[:,None,:]) was removed; verify if needed.

    # Construct filename
    if prompt_as_filename and text:
        # Convert Unicode text to ASCII (e.g., "cây cảnh" -> "cay canh") and limit to 50 characters
        filename_text = unidecode(text)[:50].replace(' ', '_')
        filename = f"{get_time()}_{filename_text}.wav"
    else:
        # Use file_path base name or default if text is empty
        base = os.path.splitext(os.path.basename(file_path))[0] if file_path else "generated"
        filename = f"{get_time()}_{base}.wav"

    # Save waveform
    save_wave(waveform, save_dir, name=filename)
    return save_dir, waveform

def parse_arguments():
    """
    Parse command-line arguments for audio generation or style transfer.

    Returns:
        Parsed arguments.
    """
    CACHE_DIR = os.getenv(
        "AUDIOLDM_CACHE_DIR",
        os.path.join(os.path.expanduser("~"), ".cache/audioldm")
    )

    parser = argparse.ArgumentParser(description="Audio generation and style transfer using AudioLDM.")

    parser.add_argument(
        "--mode",
        type=str,
        default="generation",
        help="Operation mode: 'generation' for text-to-audio, 'transfer' for style transfer.",
        choices=["generation", "transfer"]
    )

    parser.add_argument(
        "-t",
        "--text",
        type=str,
        default="",
        help="Text prompt for audio generation or style transfer."
    )

    parser.add_argument(
        "-f",
        "--file_path",
        type=str,
        default=None,
        help=("For 'transfer' mode: original audio file for style transfer. "
              "For 'generation' mode: guidance audio file for similar audio.")
    )

    parser.add_argument(
        "--transfer_strength",
        type=float,
        default=0.5,
        help="Style transfer strength (0 to 1; 0 = original, 1 = full transfer).",
    )

    parser.add_argument(
        "-s",
        "--save_dir",
        type=str,
        default="./output",
        help="Directory to save model output."
    )

    parser.add_argument(
        "--prompt_as_filename",
        action="store_true",
        help="Use text prompt as output filename."
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="audioldm-m-full",
        help="Model checkpoint to use.",
        choices=["audioldm-s-full", "audioldm-l-full", "audioldm-s-full-v2",
                 "audioldm-m-text-ft", "audioldm-s-text-ft", "audioldm-m-full"]
    )

    parser.add_argument(
        "-b",
        "--batchsize",
        type=int,
        default=1,
        help="Number of samples to generate at once."
    )

    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=200,
        help="Number of sampling steps for DDIM."
    )

    parser.add_argument(
        "-gs",
        "--guidance_scale",
        type=float,
        default=2.5,
        help="Guidance scale (larger = better quality/relevance, smaller = better diversity)."
    )

    parser.add_argument(
        "-dur",
        "--duration",
        type=float,
        default=10.0,
        help="Duration of the generated audio in seconds."
    )

    parser.add_argument(
        "-n",
        "--n_candidate_gen_per_text",
        type=int,
        default=3,
        help="Number of candidate audios to generate for quality control."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility."
    )

    return parser.parse_args()

def main():
    """Main function to run audio generation or style transfer."""
    args = parse_arguments()

    # Note: Duration constraint (multiple of 2.5) was removed; reinstate if needed.
    # assert args.duration % 2.5 == 0, "Duration must be a multiple of 2.5"

    inference(
        mode=args.mode,
        file_path=args.file_path,
        text=args.text,
        model_name=args.model_name,
        save_dir=args.save_dir,
        duration=args.duration,
        random_seed=args.seed,
        guidance_scale=args.guidance_scale,
        ddim_steps=args.ddim_steps,
        n_candidate_gen_per_text=args.n_candidate_gen_per_text,
        batchsize=args.batchsize,
        transfer_strength=args.transfer_strength,
        prompt_as_filename=args.prompt_as_filename,
    )

if __name__ == "__main__":
    main()