#!/bin/bash

# OPENAI API KEY
export OPENAI_API_KEY="sk-fake-key-for-sage"

# ANTHROPIC API KEY
export ANTHROPIC_API_KEY=""

# HUGGING FACE API KEY
export HUGGINGFACEHUB_API_TOKEN=""

# WEATHER API KEY
export OPENWEATHERMAP_API_KEY="1302937b1eb851f3e1fb7a5e080eed20"

# SMARTTHINGS API TOKEN
# Go to https://account.smartthings.com/tokens
export SMARTTHINGS_API_TOKEN=""

# Leave as empty string:
export CURL_CA_BUNDLE=""

BIN_FOLDER="$(cd "$(dirname -- "$0")" >/dev/null; pwd -P)/$(basename -- "$1")"

export SMARTHOME_ROOT="$(dirname "$BIN_FOLDER")"
export TRIGGER_SERVER_URL="0.0.0.0:5797"
export MONGODB_SERVER_URL="0.0.0.0:27017"

# Change to your own root file location
export SMARTHOME_ROOT="/home/michael/SAGE"