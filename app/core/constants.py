"""
Application Constants
Must match Flutter frontend constants
"""

# Genres - Must match EXACT order from label_encoder.json
# Order MUST match model output indices for correct predictions
GENRES = [
    "Blues",      # index 0 -> "blues"
    "Classical",  # index 1 -> "classical"
    "Country",    # index 2 -> "country"
    "Disco",      # index 3 -> "disco"
    "Hip Hop",    # index 4 -> "hiphop" 
    "Jazz",       # index 5 -> "jazz"
    "Metal",      # index 6 -> "metal"
    "Pop",        # index 7 -> "pop"
    "Reggae",     # index 8 -> "reggae"
    "Rock"        # index 9 -> "rock"
]

# Genre metadata
GENRE_METADATA = {
    "Blues": {
        "id": "blues",
        "color": "#1E88E5",
        "description": "Blues music genre"
    },
    "Classical": {
        "id": "classical",
        "color": "#536DFE",
        "description": "Classical music genre"
    },
    "Country": {
        "id": "country",
        "color": "#40C4FF",
        "description": "Country music genre"
    },
    "Disco": {
        "id": "disco",
        "color": "#FDD835",
        "description": "Disco music genre"
    },
    "Hip Hop": {
        "id": "hip_hop",
        "color": "#E040FB",
        "description": "Hip Hop music genre"
    },
    "Jazz": {
        "id": "jazz",
        "color": "#7C4DFF",
        "description": "Jazz music genre"
    },
    "Metal": {
        "id": "metal",
        "color": "#64FFDA",
        "description": "Metal music genre"
    },
    "Pop": {
        "id": "pop",
        "color": "#FF4081",
        "description": "Pop music genre"
    },
    "Reggae": {
        "id": "reggae",
        "color": "#43A047",
        "description": "Reggae music genre"
    },
    "Rock": {
        "id": "rock",
        "color": "#FF5252",
        "description": "Rock music genre"
    }
}

# Audio processing constants
AUDIO_SAMPLE_RATE = 22050
AUDIO_DURATION = 30  # seconds