# Define preset formations
def get_presets(HEIGHT, WIDTH, LOS):
    return {
    "Offense": {
        "I-Form": [
            # O - Line
            {"pos": [LOS - 20, HEIGHT // 2 - 50],  "label": "LT"},
            {"pos": [LOS - 20, HEIGHT // 2 - 25],  "label": "LG"},
            {"pos": [LOS - 20, HEIGHT // 2], "label": "C"},
            {"pos": [LOS - 20, HEIGHT // 2 + 25], "label": "RG"},
            {"pos": [LOS - 20, HEIGHT // 2 + 50],  "label": "RT"},

            # Backfield
            {"pos": [LOS - 50, HEIGHT // 2], "label": "QB"},
            {"pos": [LOS - 80, HEIGHT // 2], "label": "FB"},
            {"pos": [LOS - 110, HEIGHT // 2],"label": "HB"},

            # Receivers
            {"pos": [LOS - 20, HEIGHT // 2 - 110], "label": "WR1"},
            {"pos": [LOS - 20, HEIGHT // 2 + 135], "label": "WR2"},
            {"pos": [LOS - 30, HEIGHT // 2 + 75], "label": "TE"},
        ],
        "Singleback": [
            # O - Line
            {"pos": [LOS - 20, HEIGHT // 2 - 50],  "label": "LT"},
            {"pos": [LOS - 20, HEIGHT // 2 - 25],  "label": "LG"},
            {"pos": [LOS - 20, HEIGHT // 2], "label": "C"},
            {"pos": [LOS - 20, HEIGHT // 2 + 25], "label": "RG"},
            {"pos": [LOS - 20, HEIGHT // 2 + 50],  "label": "RT"},

            # Backfield
            {"pos": [LOS - 50, HEIGHT // 2], "label": "QB"},
            {"pos": [LOS - 110, HEIGHT // 2],"label": "HB"},

            # Receivers
            {"pos": [LOS - 20, HEIGHT // 2 - 95], "label": "WR1"},
            {"pos": [LOS - 30, HEIGHT // 2 - 75], "label": "WR2"},
            {"pos": [LOS - 30, HEIGHT // 2 + 75], "label": "TE"},
            {"pos": [LOS - 20, HEIGHT // 2 + 95], "label": "WR3"},
        ],
        "Shotgun": [
            # O - Line
            {"pos": [LOS - 20, HEIGHT // 2 - 50],  "label": "LT"},
            {"pos": [LOS - 20, HEIGHT // 2 - 25],  "label": "LG"},
            {"pos": [LOS - 20, HEIGHT // 2], "label": "C"},
            {"pos": [LOS - 20, HEIGHT // 2 + 25], "label": "RG"},
            {"pos": [LOS - 20, HEIGHT // 2 + 50],  "label": "RT"},

            # Backfield
            {"pos": [LOS - 100, HEIGHT // 2], "label": "QB"},

            # Receivers
            {"pos": [LOS - 20, HEIGHT // 2 + 100], "label": "TE"},
            {"pos": [LOS - 20, HEIGHT // 2 + 150], "label": "WR1"},
            {"pos": [LOS - 20, HEIGHT // 2 - 90], "label": "WR2"},
            {"pos": [LOS - 20, HEIGHT // 2 - 120], "label": "WR3"},
            {"pos": [LOS - 20, HEIGHT // 2 - 160],"label": "WR4"},
        ],  # Define Shotgun positions here
    },
    "Defense": {
        "4-3": [
            # Linebackers
            {"pos": [LOS + 60, HEIGHT // 2 - 20], "label": "MLB1", },
            {"pos": [LOS + 60, HEIGHT // 2 + 20],"label": "MLB2",},

            # Defensive Backs
            {"pos": [LOS + 20, HEIGHT // 2 - 110],  "label": "CB1"},
            {"pos": [LOS + 20, HEIGHT // 2 + 135], "label": "CB2",},
            {"pos": [LOS + 140, HEIGHT // 2 - 75],  "label": "FS",},
            {"pos": [LOS + 140, HEIGHT // 2 + 75],  "label": "SS",},

            # D - Line
            {"pos": [LOS + 20, HEIGHT // 2 - 55],  "label": "LOLB",},
            {"pos": [LOS + 20, HEIGHT // 2 - 25], "label": "DE1", },
            {"pos": [LOS + 20, HEIGHT // 2],  "label": "DT",},
            {"pos": [LOS + 20, HEIGHT // 2 + 25],  "label": "DE2",},
            {"pos": [LOS + 20, HEIGHT // 2 + 55], "label": "ROLB", },
        ],
        "3-4": [
            # Linebackers
            {"pos": [LOS + 60, HEIGHT // 2], "label": "MLB", },
            {"pos": [LOS + 60, HEIGHT // 2 - 40],"label": "ROLB",},
            {"pos": [LOS + 60, HEIGHT // 2 + 40],  "label": "LOLB",},

            # Defensive Backs
            {"pos": [LOS + 20, HEIGHT // 2 - 110],  "label": "CB1"},
            {"pos": [LOS + 20, HEIGHT // 2 + 135], "label": "CB2",},
            {"pos": [LOS + 140, HEIGHT // 2 - 75],  "label": "FS",},
            {"pos": [LOS + 140, HEIGHT // 2 + 75],  "label": "SS",},

            # D - Line
            {"pos": [LOS + 20, HEIGHT // 2 - 38], "label": "DE1", },
            {"pos": [LOS + 20, HEIGHT // 2 - 13],  "label": "DT1",},
            {"pos": [LOS + 20, HEIGHT // 2 + 13],  "label": "DT2",},
            {"pos": [LOS + 20, HEIGHT // 2 + 38], "label": "DE2", },
        ], 
        "Nickel": [            # Linebackers
            {"pos": [LOS + 60, HEIGHT // 2 - 40],"label": "ROLB",},
            {"pos": [LOS + 60, HEIGHT // 2 + 40],  "label": "LOLB",},

            # Defensive Backs
            {"pos": [LOS + 20, HEIGHT // 2 - 110],  "label": "CB1"},
            {"pos": [LOS + 20, HEIGHT // 2 + 135], "label": "CB2",},
            {"pos": [LOS + 20, HEIGHT // 2 - 155], "label": "CB3", },
            {"pos": [LOS + 140, HEIGHT // 2 - 75],  "label": "FS",},
            {"pos": [LOS + 140, HEIGHT // 2 + 75],  "label": "SS",},

            # D - Line
            {"pos": [LOS + 20, HEIGHT // 2 - 38], "label": "DE1", },
            {"pos": [LOS + 20, HEIGHT // 2 - 13],  "label": "DT1",},
            {"pos": [LOS + 20, HEIGHT // 2 + 13],  "label": "DT2",},
            {"pos": [LOS + 20, HEIGHT // 2 + 38], "label": "DE2", },
        ],  # Define Nickel positions here
    }}