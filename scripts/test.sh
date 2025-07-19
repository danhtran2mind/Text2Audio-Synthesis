#!/bin/bash
# Description: Example commands for using audioldm to generate and transfer audio
# Usage: Run individual commands or incorporate into a larger script
# Note: Ensure audioldm is installed and accessible in your PATH

# Exit on error, undefined variables, or pipeline failures
set -e
set -u
set -o pipefail

# --- Audio Generation Commands ---

# Generate audio from a file (default settings)
audioldm --file_path "trumpet.wav"

# Generate audio with a specified duration of 25 seconds
audioldm --file_path "trumpet.wav" --duration 25

# Generate audio with a specified duration of 2.5 seconds
audioldm --file_path "trumpet.wav" --duration 2.5

# Generate audio from a text description
audioldm --text "A hammer is hitting a wooden surface"

# Run audioldm with default/no arguments (assumes tool handles this gracefully)
audioldm

# --- Incorrect Usage Example ---
# Note: Combining --text and --file_path may ignore --text; behaves like --file_path alone
# audioldm --text "A hammer is hitting a wooden surface" --file_path "./assets/trumpet.wav"
# Warning: This is equivalent to: audioldm --file_path "./assets/trumpet.wav"

# --- Audio Transfer Command ---

# Transfer audio style to match a description
audioldm --mode "transfer" --file_path "./assets/trumpet.wav" --text "Children Singing"
