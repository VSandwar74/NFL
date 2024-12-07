# Define preset formations
def get_presets(HEIGHT, WIDTH):
    return {
    "Offense": {
        "I-Form": [
            # O - Line
            {"pos": [2 * WIDTH // 4 - 20, HEIGHT // 2 - 50],  "label": "LT"},
            {"pos": [2 * WIDTH // 4 - 20, HEIGHT // 2 - 25],  "label": "LG"},
            {"pos": [2 * WIDTH // 4 - 20, HEIGHT // 2], "label": "C"},
            {"pos": [2 * WIDTH // 4 - 20, HEIGHT // 2 + 25], "label": "RG"},
            {"pos": [2 * WIDTH // 4 - 20, HEIGHT // 2 + 50],  "label": "RT"},

            # Backfield
            {"pos": [2 * WIDTH // 4 - 50, HEIGHT // 2], "label": "QB"},
            {"pos": [2 * WIDTH // 4 - 80, HEIGHT // 2], "label": "FB"},
            {"pos": [2 * WIDTH // 4 - 110, HEIGHT // 2],"label": "HB"},

            # Receivers
            {"pos": [2 * WIDTH // 4 - 20, HEIGHT // 2 - 110], "label": "WR1"},
            {"pos": [2 * WIDTH // 4 - 20, HEIGHT // 2 + 135], "label": "WR2"},
            {"pos": [2 * WIDTH // 4 - 30, HEIGHT // 2 + 75], "label": "TE"},
        ],
        "Singleback": [
            # O - Line
            {"pos": [2 * WIDTH // 4 - 20, HEIGHT // 2 - 50],  "label": "LT"},
            {"pos": [2 * WIDTH // 4 - 20, HEIGHT // 2 - 25],  "label": "LG"},
            {"pos": [2 * WIDTH // 4 - 20, HEIGHT // 2], "label": "C"},
            {"pos": [2 * WIDTH // 4 - 20, HEIGHT // 2 + 25], "label": "RG"},
            {"pos": [2 * WIDTH // 4 - 20, HEIGHT // 2 + 50],  "label": "RT"},

            # Backfield
            {"pos": [2 * WIDTH // 4 - 50, HEIGHT // 2], "label": "QB"},
            {"pos": [2 * WIDTH // 4 - 110, HEIGHT // 2],"label": "HB"},

            # Receivers
            {"pos": [2 * WIDTH // 4 - 20, HEIGHT // 2 - 95], "label": "WR1"},
            {"pos": [2 * WIDTH // 4 - 30, HEIGHT // 2 - 75], "label": "WR2"},
            {"pos": [2 * WIDTH // 4 - 30, HEIGHT // 2 + 75], "label": "TE"},
            {"pos": [2 * WIDTH // 4 - 20, HEIGHT // 2 + 95], "label": "WR3"},
        ],
        "Shotgun": [
            # O - Line
            {"pos": [2 * WIDTH // 4 - 20, HEIGHT // 2 - 50],  "label": "LT"},
            {"pos": [2 * WIDTH // 4 - 20, HEIGHT // 2 - 25],  "label": "LG"},
            {"pos": [2 * WIDTH // 4 - 20, HEIGHT // 2], "label": "C"},
            {"pos": [2 * WIDTH // 4 - 20, HEIGHT // 2 + 25], "label": "RG"},
            {"pos": [2 * WIDTH // 4 - 20, HEIGHT // 2 + 50],  "label": "RT"},

            # Backfield
            {"pos": [2 * WIDTH // 4 - 100, HEIGHT // 2], "label": "QB"},

            # Receivers
            {"pos": [2 * WIDTH // 4 - 20, HEIGHT // 2 + 100], "label": "TE"},
            {"pos": [2 * WIDTH // 4 - 20, HEIGHT // 2 + 150], "label": "WR1"},
            {"pos": [2 * WIDTH // 4 - 20, HEIGHT // 2 - 90], "label": "WR2"},
            {"pos": [2 * WIDTH // 4 - 20, HEIGHT // 2 - 120], "label": "WR3"},
            {"pos": [2 * WIDTH // 4 - 20, HEIGHT // 2 - 160],"label": "WR4"},
        ],  # Define Shotgun positions here
    },
    "Defense": {
        "4-3": [
            # Linebackers
            {"pos": [2 * WIDTH // 4 + 60, HEIGHT // 2 - 20], "label": "MLB1", },
            {"pos": [2 * WIDTH // 4 + 60, HEIGHT // 2 + 20],"label": "MLB2",},

            # Defensive Backs
            {"pos": [2 * WIDTH // 4 + 20, HEIGHT // 2 - 110],  "label": "CB1"},
            {"pos": [2 * WIDTH // 4 + 20, HEIGHT // 2 + 135], "label": "CB2",},
            {"pos": [2 * WIDTH // 4 + 140, HEIGHT // 2 - 75],  "label": "FS",},
            {"pos": [2 * WIDTH // 4 + 140, HEIGHT // 2 + 75],  "label": "SS",},

            # D - Line
            {"pos": [2 * WIDTH // 4 + 20, HEIGHT // 2 - 55],  "label": "LOLB",},
            {"pos": [2 * WIDTH // 4 + 20, HEIGHT // 2 - 25], "label": "DE1", },
            {"pos": [2 * WIDTH // 4 + 20, HEIGHT // 2],  "label": "DT",},
            {"pos": [2 * WIDTH // 4 + 20, HEIGHT // 2 + 25],  "label": "DE2",},
            {"pos": [2 * WIDTH // 4 + 20, HEIGHT // 2 + 55], "label": "ROLB", },
        ],
        "3-4": [
            # Linebackers
            {"pos": [2 * WIDTH // 4 + 60, HEIGHT // 2], "label": "MLB", },
            {"pos": [2 * WIDTH // 4 + 60, HEIGHT // 2 - 40],"label": "ROLB",},
            {"pos": [2 * WIDTH // 4 + 60, HEIGHT // 2 + 40],  "label": "LOLB",},

            # Defensive Backs
            {"pos": [2 * WIDTH // 4 + 20, HEIGHT // 2 - 110],  "label": "CB1"},
            {"pos": [2 * WIDTH // 4 + 20, HEIGHT // 2 + 135], "label": "CB2",},
            {"pos": [2 * WIDTH // 4 + 140, HEIGHT // 2 - 75],  "label": "FS",},
            {"pos": [2 * WIDTH // 4 + 140, HEIGHT // 2 + 75],  "label": "SS",},

            # D - Line
            {"pos": [2 * WIDTH // 4 + 20, HEIGHT // 2 - 38], "label": "DE1", },
            {"pos": [2 * WIDTH // 4 + 20, HEIGHT // 2 - 13],  "label": "DT1",},
            {"pos": [2 * WIDTH // 4 + 20, HEIGHT // 2 + 13],  "label": "DT2",},
            {"pos": [2 * WIDTH // 4 + 20, HEIGHT // 2 + 38], "label": "DE2", },
        ], 
        "Nickel": [],  # Define Nickel positions here
    }}