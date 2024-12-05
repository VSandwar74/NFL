# TODO
'''
- play dropdown
    - defensive alignments/adjustments
- camera angle
'''

import pygame
import sys
import math

# Initialize Pygame
pygame.init()

# Screen dimensions and setup
WIDTH, HEIGHT = 1000, 500
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("American Football Formation")

# Colors
GREEN = (34, 139, 34)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
GREY = (200, 200, 200)
YELLOW = (255, 255, 0)

# Circle properties
CIRCLE_RADIUS = 10  # Scaled-down size for players
VECTOR_LENGTH = 50  # Default length of velocity vectors

# Formations
def get_red_offense():
    """Returns positions for the offense in an I-formation."""
    positions = [
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
        {"pos": [2 * WIDTH // 4 - 20, HEIGHT // 2 + 75], "label": "TE"},
    ]
    
    return [{"pos": pos["pos"], "label": pos["label"], "color": RED, "dragging": False, "vector": [0, 0]} for pos in positions]

def get_blue_defense():
    """Returns positions for the defense in a 2-high man shell."""
    positions = [
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
    ]

    return [{"pos": pos["pos"], "label": pos["label"], "color": BLUE, "dragging": False, "vector": [0, 0]} for pos in positions]

# Combine all players
circles = get_red_offense() + get_blue_defense()

# Draw American football field
def draw_field():
    screen.fill(GREEN)
    pygame.draw.rect(screen, WHITE, (50, 50, WIDTH - 100, HEIGHT - 100), 5)
    for i in range(1, 10):
        x = 50 + i * (WIDTH - 100) // 10
        pygame.draw.line(screen, WHITE, (x, 50), (x, HEIGHT - 50), 2)
    pygame.draw.rect(screen, WHITE, (50, 50, 50, HEIGHT - 100), 5)
    pygame.draw.rect(screen, WHITE, (WIDTH - 100, 50, 50, HEIGHT - 100), 5)
    pygame.draw.line(screen, WHITE, (WIDTH // 2, 50), (WIDTH // 2, HEIGHT - 50), 5)

# Draw toggle button
def draw_toggle():
    pygame.draw.rect(screen, GREY, (WIDTH // 2 - 50, 10, 100, 30), border_radius=10)
    font = pygame.font.Font(None, 24)
    text = font.render(mode, True, BLUE)
    screen.blit(text, (WIDTH // 2 - 30, 17))

# Draw velocity vector
def draw_vector(circle):
    start_pos = circle["pos"]
    end_pos = [start_pos[0] + circle["vector"][0], start_pos[1] + circle["vector"][1]]
    pygame.draw.line(screen, BLACK, start_pos, end_pos, 2)
    pygame.draw.circle(screen, BLACK, end_pos, 5)

# Draw play button
def draw_play_button():
    pygame.draw.circle(screen, YELLOW, (WIDTH - 70, 30), 20)
    pygame.draw.polygon(screen, BLACK, [(WIDTH - 80, 20), (WIDTH - 60, 30), (WIDTH - 80, 40)])


# Update player positions
def update_positions(elapsed_time):
    for circle in circles:
        vector = circle["vector"]
        if vector[0] == 0 and vector[1] == 0:
            continue
        # Calculate movement step
        start_pos = circle["pos"]
        move_x = vector[0] * elapsed_time
        move_y = vector[1] * elapsed_time
        circle["pos"][0] += move_x
        circle["pos"][1] += move_y
        # Gradually decrease velocity
        circle["vector"][0] *= (1 - elapsed_time)
        circle["vector"][1] *= (1 - elapsed_time)

# Variables
active_position = "Cursor"
mode = "Position"  # Default mode
playing = False
start_time = None

# Main loop
running = True
clock = pygame.time.Clock()

while running:
    current_time = pygame.time.get_ticks() / 1000  # Time in seconds
    elapsed_time = 0
    if playing and start_time:
        elapsed_time = current_time - start_time
        start_time = current_time
        update_positions(elapsed_time)
        # Stop animation after 1 second
        if elapsed_time >= 1:
            playing = False
            start_time = None

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            # Toggle mode button
            if WIDTH // 2 - 50 <= event.pos[0] <= WIDTH // 2 + 50 and 10 <= event.pos[1] <= 50:
                mode = "Vector" if mode == "Position" else "Position"
            # Play button
            elif (event.pos[0] - (WIDTH - 70)) ** 2 + (event.pos[1] - 30) ** 2 <= 20 ** 2:
                if not playing:
                    playing = True
                    start_time = current_time
            else:
                for circle in circles:
                    dx = event.pos[0] - circle["pos"][0]
                    dy = event.pos[1] - circle["pos"][1]
                    if dx * dx + dy * dy <= CIRCLE_RADIUS * CIRCLE_RADIUS:
                        circle["dragging"] = True
        elif event.type == pygame.MOUSEBUTTONUP:
            for circle in circles:
                circle["dragging"] = False
        elif event.type == pygame.MOUSEMOTION:
            for circle in circles:
                if circle["dragging"]:
                    if mode == "Position":
                        new_x, new_y = event.pos
                        if circle["color"] == RED and new_x < WIDTH // 2:
                            circle["pos"] = [new_x, new_y]
                        elif circle["color"] == BLUE and new_x > WIDTH // 2:
                            circle["pos"] = [new_x, new_y]
                    elif mode == "Vector":
                        start_pos = circle["pos"]
                        circle["vector"] = [event.pos[0] - start_pos[0], event.pos[1] - start_pos[1]]

    draw_field()
    draw_toggle()
    draw_play_button()

    cursor_pos = pygame.mouse.get_pos()
    font = pygame.font.Font(None, 24)
    text = font.render(f"{active_position}: {cursor_pos}", True, BLACK)
    screen.blit(text, (10, 10))


    # Draw players
    font = pygame.font.Font(None, 24)
    for circle in circles:
        pygame.draw.circle(screen, circle["color"], circle["pos"], CIRCLE_RADIUS)
        dx = pygame.mouse.get_pos()[0] - circle["pos"][0]
        dy = pygame.mouse.get_pos()[1] - circle["pos"][1]
        if dx * dx + dy * dy <= CIRCLE_RADIUS * CIRCLE_RADIUS:
            label_text = font.render(f"{circle['label']}", True, BLACK)
            active_position = circle['label']
            screen.blit(label_text, (circle["pos"][0] + 15, circle["pos"][1] - 15))
        draw_vector(circle)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()
