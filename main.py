# TODO
'''
- play dropdown
    - defensive alignments/adjustments
- camera angle
'''

import pygame
import sys
import math
from pygame.locals import *
from formations import get_presets
import torch
from model import model, prepare_tensor
import pandas as pd

# Initialize Pygame
pygame.init()

# Screen dimensions and setup
WIDTH, HEIGHT = 1180, 580 # true dimensions are 120x53.3 scaled up to 1080x480, plus 50 px on each side
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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


current_formation = {"Offense": "Singleback", "Defense": "3-4"}

model.load_state_dict(torch.load('./best_model_week3.pth', weights_only=True, map_location=DEVICE))
model.eval()

# Formations
def get_red_offense():
    """Returns positions for the offense in an I-formation."""
    positions = get_presets(HEIGHT, WIDTH)["Offense"][current_formation["Offense"]]
    return [{"pos": pos["pos"], "label": pos["label"], "color": RED, "dragging": False, "vector": [0, 0]} for pos in positions]

def get_blue_defense():
    """Returns positions for the defense in a 2-high man shell."""
    positions = get_presets(HEIGHT, WIDTH)["Defense"][current_formation["Defense"]]
    [
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

def create_player_dataframe(circles):
  """
  Creates a pandas DataFrame from the list of player positions and vectors.

  Args:
      circles: A list of dictionaries representing player positions and vectors.

  Returns:
      A pandas DataFrame with columns:
          x_clean: Cleaned x-position (accounting for field boundaries).
          y_clean: Cleaned y-position (accounting for field boundaries).
          v_x: x-component of the velocity vector.
          v_y: y-component of the velocity vector.
          defense: 0 if offensive player, 1 if defensive player.
  """
  data = []
  for circle in circles:
    # Adjust positions based on team color and field boundaries
    x_pos = circle["pos"][0]
    y_pos = circle["pos"][1]
    if circle["color"] == RED:
      x_pos = max(x_pos, 50)  # Limit offense to their side of the field
    else:
      x_pos = min(x_pos, WIDTH - 50)  # Limit defense to their side of the field

    data.append({
        "frameId": 1,
        "x_clean": x_pos,
        "y_clean": min(max(y_pos, 50), HEIGHT - 50),  # Clamp y-position to field bounds
        "v_x": circle["vector"][0],
        "v_y": circle["vector"][1],
        "defense": 1 if circle["color"] == BLUE else 0  # 1 for defense, 0 for offense
    })

  return pd.DataFrame(data)

def predict_coverage(circles):
  """
  Prepares the positions data as a tensor and predicts zone/man coverage.

  Args:
      positions: A representation of the current player positions.

  Returns:
      zone_prob: Probability of zone coverage.
      man_prob: Probability of man coverage.
  """
  # Prepare positions data as a tensor (replace with your specific logic)
  positions = create_player_dataframe(circles)
  frame_tensor = prepare_tensor(positions)

  frame_tensor = frame_tensor.to(DEVICE)  # Move to device if necessary

  with torch.no_grad():
      outputs = model(frame_tensor)  # Shape: [num_frames, num_classes]
      probabilities = torch.softmax(outputs, dim=1).cpu().numpy()

      zone_prob = probabilities[0][0]
      man_prob = probabilities[0][1]

  return zone_prob, man_prob

# Draw American football field
def draw_field():
    screen.fill(GREEN)
    pygame.draw.rect(screen, WHITE, (50, 50, WIDTH - 100, HEIGHT - 100), 5)

    # Yardage Lines
    for i in range(1, 10):
        x = 140 + i * (WIDTH - 280) // 10
        pygame.draw.line(screen, WHITE, (x, 50), (x, HEIGHT - 50), 2)

    # Left and Right TD
    pygame.draw.rect(screen, RED, (50, 50, 90, HEIGHT - 100))
    pygame.draw.rect(screen, WHITE, (50, 50, 90, HEIGHT - 100), 5)
    pygame.draw.rect(screen, BLUE, (WIDTH - 140, 50, 90, HEIGHT - 100))
    pygame.draw.rect(screen, WHITE, (WIDTH - 140, 50, 90, HEIGHT - 100), 5)

    # 50 yd line
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

field_orientation = "horizontal"  # Default to horizontal

def flip_orientation():
    global field_orientation
    field_orientation = "vertical" if field_orientation == "horizontal" else "horizontal"


def draw_dropdown_menu():
    # Render dropdown menu for formations
    font = pygame.font.Font(None, 30)
    y_offset = 10
    for team, formations in get_presets(HEIGHT, WIDTH).items():
        text_surface = font.render(f"{team}: {current_formation[team]}", True, BLACK)
        screen.blit(text_surface, (10, y_offset))
        y_offset += 30

def set_formation(team, formation):
    global current_formation, circles
    current_formation[team] = formation
    circles = get_red_offense() + get_blue_defense()
    # positions = get_presets(HEIGHT, WIDTH)[team][formation]

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

            # print(circles)
            
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
                        # Boundaries
                        if 50 < new_x < WIDTH - 50 and 50 < new_y < HEIGHT - 50:
                            # LOS Bounds
                            if circle["color"] == RED and new_x < WIDTH // 2:
                                circle["pos"] = [new_x, new_y]
                            elif circle["color"] == BLUE and new_x > WIDTH // 2:
                                circle["pos"] = [new_x, new_y]
                    elif mode == "Vector":
                        start_pos = circle["pos"]
                        circle["vector"] = [event.pos[0] - start_pos[0], event.pos[1] - start_pos[1]]
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_1:  
                set_formation("Offense", "I-Form")
            elif event.key == pygame.K_2:  
                set_formation("Offense", "Singleback")
            elif event.key == pygame.K_3:  
                set_formation("Offense", "Shotgun")
            elif event.key == pygame.K_8:
                set_formation("Defense", "4-3")
            elif event.key == pygame.K_9:
                set_formation("Defense", "3-4")

    draw_field()
    draw_toggle()
    draw_play_button()

    # Get coverage prediction
    zone, man = predict_coverage(circles)
    font = pygame.font.Font(None, 24)
    text = font.render(f"Zone: {zone:.2f}, Man: {man:.2f}", True, BLACK)
    screen.blit(text, (WIDTH // 2 - 75, HEIGHT - 30))

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
