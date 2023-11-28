import pandas as pd

genre_normalization_map = {
    "francophone": "french",
    "francais": "french",
    "french": "french",
    "chanson": "french",
    "rock": "rock",
    "mellow gold": "rock",
    "neo mellow": "rock",
    "metal": "metal",
    "country": "country",
    "hip hop": "rap",
    "rap": "rap",
    "hiphop": "rap",
    "hip-hop": "rap",
    "urban": "rap",
    "trap": "rap",
    "bboy": "rap",
    "drill": "rap",
    "pop": "pop",
    "new wave": "pop",
    "anime": "japanese",
    "japanese": "japanese",
    "emo": "emo",
    "punk": "punk",
    "screamo": "emo",
    "hardcore": "emo",
    "dreamo": "emo",
    "grunge": "grunge",
    "permanent wave": "emo",
    "new wave": "emo",
    "lofi": "lofi",
    "lo-fi": "lofi",
    "chill": "lofi",
    "chillhop": "lofi",
    "chillout": "lofi",
    "chillwave": "lofi",
    "room": "lofi",
    "indie": "indie",
    "edm": "electronic",
    "electro": "electronic",
    "electronic": "electronic",
    "house": "electronic",
    "techno": "electronic",
    "dubstep": "electronic",
    "trance": "electronic",
    "r&b": "rnb",
    "soul": "rnb",
    "rnb": "rnb",
    "jazz": "jazz/blues",
    "blues": "jazz/blues",
    "classical": "classical/opera",
    "opera": "classical/opera",
    "uk": "english",
    "british": "english",
    "latino": "spanish",
    "spanish": "spanish",
    "latin": "spanish",
    "afro": "afro",
}

def normalize_genres(genres: pd.Series) -> pd.Series:
    """
    Normalize genre names to a standard set of genres.
    :param genres: A series of lists of genres List[str].
    :return: A series of lists of normalized genres List[str].
    """
    genres = genres.copy()

    def normalize_genre_list(genre_list: list) -> list:
        for i, genre in enumerate(genre_list):
            # check if genre contains a part of a genre in the map
            for key in genre_normalization_map.keys():
                if key in genre:
                    genre_list[i] = genre_normalization_map[key]

        return genre_list
    
    return genres.apply(normalize_genre_list)