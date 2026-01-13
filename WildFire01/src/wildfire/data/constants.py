DATA_SIZE = 64

ENV11 = [
    "elevation", "th", "vs", "tmmn", "tmmx",
    "sph", "pr", "pdsi", "NDVI", "population", "erc"
]
PREV_KEY = "PrevFireMask"
NEXT_KEY = "FireMask"

INPUT_FEATURES  = ENV11 + [PREV_KEY]
OUTPUT_FEATURES = [NEXT_KEY]
C_IN = len(INPUT_FEATURES)
