"""Generate audio from text using the AudioLDM model and save the output as a WAV file."""

import os
import argparse

from audioldm import text_to_audio, build_model, save_wave


def parse_arguments():
    """Parse command-line arguments for audio generation."""
    parser = argparse.ArgumentParser(description="Generate audio from text using AudioLDM.")
    
    parser.add_argument(
        "-t",
        "--text",
        type=str,
        default="A hammer is hitting a wooden surface",
        help="Text prompt for audio generation.",
    )
    
    parser.add_argument(
        "-s",
        "--save_path",
        type=str,
        default="./output",
        help="Path to save the generated audio output.",
    )
    
    parser.add_argument(
        "-ckpt",
        "--ckpt_path",
        type=str,
        default="./ckpt/audioldm-s-full.ckpt",
        help="Path to the pretrained .ckpt model file.",
    )
    
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=1,
        help="Number of samples to generate simultaneously.",
    )
    
    parser.add_argument(
        "-gs",
        "--guidance_scale",
        type=float,
        default=2.5,
        help=(
            "Guidance scale: higher values improve quality and text relevance; "
            "lower values increase diversity."
        ),
    )
    
    parser.add_argument(
        "-dur",
        "--duration",
        type=float,
        default=10.0,
        help="Duration of the generated audio in seconds (must be a multiple of 2.5).",
    )
    
    parser.add_argument(
        "-n",
        "--n_candidates_per_text",
        type=int,
        default=3,
        help=(
            "Number of candidate audios to generate per text for quality control. "
            "Higher values improve quality but increase computation."
        ),
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for generation (any integer; changes output).",
    )
    
    return parser.parse_args()


def main():
    """Main function to generate and save audio from text input."""
    args = parse_arguments()
    
    # Validate duration
    if args.duration % 2.5 != 0:
        raise ValueError("Duration must be a multiple of 2.5 seconds")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.save_path, exist_ok=True)
    
    # Build model and generate audio
    audioldm = build_model(ckpt_path=args.ckpt_path)
    waveform = text_to_audio(
        audioldm,
        text=args.text,
        seed=args.seed,
        duration=args.duration,
        guidance_scale=args.guidance_scale,
        n_candidate_gen_per_text=args.n_candidates_per_text,
        batchsize=args.batch_size,
    )
    
    # Save the generated waveform
    save_wave(waveform, args.save_path, name=args.text)


if __name__ == "__main__":
    main()
