import json
from typing import List
import numpy as np
import pygame
import sys
import math
from pygame.locals import *
from formations import get_presets
import torch
from model import model, prepare_tensor
import pandas as pd
import asyncio
import random
import requests

# TODO
# - Add horizontal/vertical swap


# Initialize Pygame
pygame.init()

# Screen dimensions and setup
WIDTH, HEIGHT = 1330, 580  # 150 px for dropdown menu
FIELD_WIDTH = WIDTH - 150 # true dimensions are 120x53.3 scaled up to 1080x480, plus 50 px on each side
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("American Football Formation")
API_URL = "https://nfl-f1ma.onrender.com"

# Colors
GREEN = (34, 139, 34)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
YELLOW = (255, 255, 0)

current_formation = {"Offense": "Singleback", "Defense": "3-4"}

# Circle properties
CIRCLE_RADIUS = 10  # Scaled-down size for players
VECTOR_LENGTH = 50  # Default length of velocity vectors
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Formations
def get_red_offense(los_x):
    """Returns positions for the offense in an I-formation."""
    if current_formation["Offense"] == 'Custom':
        return circles
    positions = get_presets(HEIGHT, FIELD_WIDTH, los_x)["Offense"][current_formation["Offense"]]
    return [{"pos": pos["pos"], "label": pos["label"], "color": RED, "dragging": False, "vector": [0, 0]} for pos in positions]

def get_blue_defense(los_x):
    """Returns positions for the defense in a 2-high man shell."""
    if current_formation["Defense"] == 'Custom':
        return circles  
    positions = get_presets(HEIGHT, FIELD_WIDTH, los_x)["Defense"][current_formation["Defense"]]
    return [{"pos": pos["pos"], "label": pos["label"], "color": BLUE, "dragging": False, "vector": [0, 0]} for pos in positions]

# Combine all players
# circles = get_red_offense() + get_blue_defense()

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
        # if circle["color"] == RED:
        #   x_pos = max(x_pos, 50)  # Limit offense to their side of the field
        # else:
        #   x_pos = min(x_pos, WIDTH - 50)  # Limit defense to their side of the field

        data.append({
            "frameId": 1,
            "x_clean": (x_pos - 50) / 9,
            "y_clean": (y_pos - 50) / 9,# min(max(y_pos, 50), HEIGHT - 50),  # Clamp y-position to field bounds
            "v_x": circle["vector"][0] / 9,
            "v_y": circle["vector"][1] / 9,
            "defense": 1 if circle["color"] == BLUE else 0  # 1 for defense, 0 for offense
        })

    return pd.DataFrame(data)

def draw_play_dropdown():
    font = pygame.font.Font(None, 24)
    y_offset = 70  # Start dropdown below toggle
    for i, formation in enumerate(["3-4", "4-3", "Nickel"]):
        rect = pygame.Rect(1180, y_offset, 100, 25)
        pygame.draw.rect(screen, GRAY if current_formation["Defense"] != formation else YELLOW, rect)

        text_surface = font.render(formation, True, BLACK)
        text_rect = text_surface.get_rect(center=rect.center)  # Center the text rect
        screen.blit(text_surface, text_rect)

        y_offset += 30
    y_offset += 75
    for i, formation in enumerate(["I-Form", "Singleback", "Shotgun"]):
        rect = pygame.Rect(1180, y_offset, 100, 25)
        pygame.draw.rect(screen, GRAY if current_formation["Offense"] != formation else YELLOW, rect)

        text_surface = font.render(formation, True, BLACK)
        text_rect = text_surface.get_rect(center=rect.center)  # Center the text rect
        screen.blit(text_surface, text_rect)

        y_offset += 30
    y_offset += 75
    for i, formation in enumerate(["CAR v. CIN", "GB v. DET"]):
        rect = pygame.Rect(1180, y_offset, 100, 25)
        pygame.draw.rect(screen, GRAY if current_formation["Offense"] != formation else YELLOW, rect)

        text_surface = font.render(formation, True, BLACK)
        text_rect = text_surface.get_rect(center=rect.center)  # Center the text rect
        screen.blit(text_surface, text_rect)

        y_offset += 30
    

def handle_dropdown_click(pos, los_x):
    y_offset = 70
    for i, formation in enumerate(["3-4", "4-3", "Nickel"]):
        if y_offset <= pos[1] <= y_offset + 25:
            return set_formation("Defense", formation, los_x), los_x
        y_offset += 30
    y_offset += 75
    for i, formation in enumerate(["I-Form", "Singleback", "Shotgun"]):
        if y_offset <= pos[1] <= y_offset + 25:
            return set_formation("Offense", formation, los_x), los_x
        y_offset += 30
    y_offset += 75
    for i, formation in enumerate(["CAR v. CIN", "GB v. DET"]):
        if y_offset <= pos[1] <= y_offset + 25:
            los, direction = None, None
            if formation == "CAR v. CIN": 
                with open('cincy.json', 'r') as f:
                    data = json.load(f)
            elif formation == "GB v. DET":
                with open('pack.json', 'r') as f:
                    data = json.load(f)
            df = pd.DataFrame(data)
            print(df)
            for idx, row in df.iterrows():
                if row['club'] == 'football':
                    los = row['x']
                    direction = row['playDirection']
                    break
            circles = []
            for idx, row in df.iterrows():
                if row.club == 'football':
                    continue
                circles.append({
                    "pos": [round(row.x * 9 + 50, 2), round(row.y * 9 + 50, 2)],
                    "label": row.displayName.split(' ')[-1],
                    "color": (0, 0, 255) if (direction == 'left' and row['x'] < los) or (direction == 'right' and row['x'] > los) else (255, 0, 0),
                    "dragging": False,
                    "vector": [(row.s * 9) * math.cos(float(row.dir) - 90), (row.s * 9) * math.sin(float(row.dir) - 90)],
                })
            # use_preset_positions(circles, los * 9 + 50)
            return circles, los * 9 + 50
        y_offset += 30

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
def draw_rect_alpha(surface, color, rect):
    shape_surf = pygame.Surface(pygame.Rect(rect).size, pygame.SRCALPHA)
    pygame.draw.rect(shape_surf, color, shape_surf.get_rect())
    surface.blit(shape_surf, rect)

def draw_circle_alpha(surface, color, center, radius):
    target_rect = pygame.Rect(center, (0, 0)).inflate((radius * 2, radius * 2))
    shape_surf = pygame.Surface(target_rect.size, pygame.SRCALPHA)
    pygame.draw.circle(shape_surf, color, (radius, radius), radius)
    surface.blit(shape_surf, target_rect)

def get_noise(density=0.01):  # Density controls the number of dots
    noise_dots = []
    area = (FIELD_WIDTH - 100) * (HEIGHT - 100)  # Usable area
    num_dots = int(area * density)

    # Calculate an approximate spacing based on the number of dots
    spacing = math.sqrt(area / num_dots)

    for i in range(num_dots):
        #Distribute the dots somewhat evenly
        x = int(50 + (i % int((FIELD_WIDTH - 100) / spacing)) * spacing + random.uniform(-spacing/4,spacing/4))
        y = int(50 + (i // int((FIELD_WIDTH - 100) / spacing)) * spacing + random.uniform(-spacing/4,spacing/4))

        color = random.choice([(255, 255, 255, 50), (0, 0, 0, 50)])
        noise_dots.append((x, y, color))

    return noise_dots

def draw_noise_dots(surface, noise_dots):  # Function to draw noise dots once
    for x, y, color in noise_dots:
        draw_circle_alpha(surface, color, (x, y), 1)

def draw_field(dots, los_x):
    light_green = (54, 125, 8)  # Lighter green
    dark_green = (49, 108, 6)  # Darker green
    
    screen.fill(light_green)

    # Alternate between lighter and darker green for every 10-yard segment
    for i in range(10):
        if i % 2 == 0:
            pygame.draw.rect(screen, light_green, (140 + i * (FIELD_WIDTH - 280) // 10, 50, (FIELD_WIDTH - 140) // 10, HEIGHT - 100))
        else:
            pygame.draw.rect(screen, dark_green, (140 + i * (FIELD_WIDTH - 280) // 10, 50, (FIELD_WIDTH - 140) // 10, HEIGHT - 100))
    
    # # 1-pixel stripes slightly off-color across the field
    # stripe_color = (30, 130, 30)  # Slightly darker green
    # for y in range(50, HEIGHT - 50, 5):  # Spaced 5 pixels apart
    #     pygame.draw.line(screen, stripe_color, (50, y), (FIELD_WIDTH - 50, y), 1)

    # Draw transparent stripes using draw_rect_alpha
    stripe_color = (30, 130, 30, 128)  # Slightly darker green with alpha (128 for 50% transparency)
    for y in range(50, HEIGHT - 50, 5):  # Spaced 5 pixels apart
        rect = (50, y, FIELD_WIDTH - 100, 2) #Create a rectangle for the line
        draw_rect_alpha(screen, stripe_color, rect)
    
    # Left and Right TD areas
    pygame.draw.rect(screen, BLUE, (50, 50, 90, HEIGHT - 100))  # Left TD
    pygame.draw.rect(screen, WHITE, (50, 50, 90, HEIGHT - 100), 5)

    pygame.draw.rect(screen, RED, (FIELD_WIDTH - 140, 50, 90, HEIGHT - 100))  # Right TD
    pygame.draw.rect(screen, WHITE, (FIELD_WIDTH - 140, 50, 90, HEIGHT - 100), 5)

    # Add random noise dots for texture
    draw_noise_dots(screen, dots)

    # Draw white field outline
    pygame.draw.rect(screen, WHITE, (50, 50, FIELD_WIDTH - 100, HEIGHT - 100), 5)

    # Yardage lines
    for i in range(1, 10):
        x = 140 + i * (FIELD_WIDTH - 100) // 12
        pygame.draw.line(screen, WHITE, (x, 50), (x, HEIGHT - 50), 2)

    # 50-yard line
    pygame.draw.line(screen, WHITE, (FIELD_WIDTH // 2, 50), (FIELD_WIDTH // 2, HEIGHT - 50), 5)

    # Draw dotted line effect
    for y in range(30, HEIGHT - 30, 10):  # Adjust spacing for dot density
        pygame.draw.line(screen, YELLOW, (los_x, y), (los_x, y + 5), 3)  # Draw short line segments

# Draw toggle button
def draw_toggle(mode):
    pygame.draw.rect(screen, GRAY, (WIDTH // 2 - 50, 10, 100, 30), border_radius=10)
    font = pygame.font.Font(None, 24)
    text = font.render(mode, True, BLUE)
    screen.blit(text, (WIDTH // 2 - 30, 17))

def update_los(circles: List[dict], los_x_offset):
    for circle in circles:
        circle["pos"][0] += los_x_offset

# Draw velocity vector
def draw_vector(circle):
    start_pos = circle["pos"]
    end_pos = [start_pos[0] + circle["vector"][0], start_pos[1] + circle["vector"][1]]
    pygame.draw.line(screen, BLACK, start_pos, end_pos, 2)
    pygame.draw.circle(screen, BLACK, end_pos, 5)

# Draw play button
def draw_play_button(playing):
    pygame.draw.circle(screen, YELLOW, (WIDTH - 70, 30), 20)
    if not playing:
        pygame.draw.polygon(screen, BLACK, [(WIDTH - 80, 20), (WIDTH - 60, 30), (WIDTH - 80, 40)])
    else:
        pygame.draw.rect(screen, BLACK, (WIDTH - 80, 20, 5, 20))
        pygame.draw.rect(screen, BLACK, (WIDTH - 65, 20, 5, 20))

field_orientation = "horizontal"  # Default to horizontal
def flip_orientation():
    global field_orientation
    field_orientation = "vertical" if field_orientation == "horizontal" else "horizontal"

def draw_input_box(input_box_active, input_text):
    global input_box, button_rect
    # Input box
    input_box = pygame.Rect(1155, HEIGHT - 100, 150, 30)

    # Button
    button_rect = pygame.Rect(1205, HEIGHT - 50, 50, 30)

    # Draw the input box
    if input_box_active:
        pygame.draw.rect(screen, GRAY, input_box, 2)  # Draw a thicker border for focus
    else:
        pygame.draw.rect(screen, BLUE, input_box)
    text_surface = pygame.font.Font(None, 25).render(input_text, True, BLACK)
    text_rect = text_surface.get_rect(center=input_box.center)
    screen.blit(text_surface, text_rect)

    # Draw the button
    pygame.draw.rect(screen, GRAY, button_rect)
    button_text = pygame.font.Font(None, 25).render("Send", True, BLACK)
    button_text_rect = button_text.get_rect(center=button_rect.center)
    screen.blit(button_text, button_text_rect)

# def draw_dropdown_menu():
#     # Render dropdown menu for formations
#     font = pygame.font.Font(None, 30)
#     y_offset = 10
#     for team, formations in get_presets(HEIGHT, WIDTH).items():
#         text_surface = font.render(f"{team}: {current_formation[team]}", True, BLACK)
#         screen.blit(text_surface, (10, y_offset))
#         y_offset += 30
def handle_api_request(input_text, input_box_active):
    gameId, playId = input_text.split('_')
    endpoint = f"{API_URL}/tracking/?gameId={gameId}&playId={playId}"
    try:
        response = requests.get(endpoint)
        response.raise_for_status()  # Raise an exception for bad status codes
        data = response.json()  # Print the API response (if applicable)
    except requests.exceptions.RequestException as e:
        print(f"Error sending data to API: {e}")
    input_text = ''  # Clear the input box after sending
    input_box_active = False  # Deactivate the input box

    los, direction = None, None
    df = pd.DataFrame(data['data'])
    print(df)
    for idx, row in df.iterrows():
        if row['club'] == 'football':
            los = row['x']
            direction = row['playDirection']
            break
    circles = []
    for idx, row in df.iterrows():
        if row.club == 'football':
            continue
        circles.append({
            "pos": [round(row.x * 9 + 50, 2), round(row.y * 9 + 50, 2)],
            "label": row.displayName.split(' ')[-1],
            "color": (0, 0, 255) if (direction == 'left' and row['x'] < los) or (direction == 'right' and row['x'] > los) else (255, 0, 0),
            "dragging": False,
            "vector": [(row.s * 9) * math.cos(float(row.dir) - 90), (row.s * 9) * math.sin(float(row.dir) - 90)],
        })
    use_preset_positions(circles, los * 9 + 50)

def set_formation(team, formation, los_x):
    global current_formation
    current_formation[team] = formation
    circles = get_red_offense(los_x) + get_blue_defense(los_x)
    return circles
    # positions = get_presets(HEIGHT, WIDTH)[team][formation]

def use_preset_positions(preset, new_los):
    global current_formation, circles, los_x
    current_formation["Offense"] = 'Custom'
    current_formation["Defense"] = 'Custom'
    los_x = new_los
    circles = preset

# Update player positions
def update_positions(elapsed_time, circles):
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

def main():

    # Variables
    active_position = "Cursor"
    mode = "Position"  # Default mode
    playing = False
    start_time = None
    noise = 0
    dots = []
    dragging_los = False # Variable to track if the LOS is being dragged
    los_start_x = 0 # store the start x position of the mouse click
    input_text = ''
    input_box_active = False
    los_x = FIELD_WIDTH // 2  # Initial LOS position

    circles = get_red_offense(los_x) + get_blue_defense(los_x)

    model.load_state_dict(torch.load('./best_model_week3.pth', weights_only=True, map_location=DEVICE))
    model.eval()

    # Main loop
    running = True
    clock = pygame.time.Clock()

    while running:
        current_time = pygame.time.get_ticks() / 1000  # Time in seconds
        elapsed_time = 0

        if playing and start_time:
            elapsed_time = current_time - start_time
            start_time = current_time
            update_positions(elapsed_time, circles)
            # Stop animation after 1 second
            if elapsed_time >= 1:
                playing = False
                start_time = None

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # print(circles)
                if 1180 <= event.pos[0] <= WIDTH and 70 <= event.pos[1] <= HEIGHT:
                    circles, los_x = handle_dropdown_click(event.pos, los_x)
                # Toggle mode button
                elif WIDTH // 2 - 50 <= event.pos[0] <= WIDTH // 2 + 50 and 10 <= event.pos[1] <= 50:
                    mode = "Vector" if mode == "Position" else "Position"
                # Play button
                elif (event.pos[0] - (WIDTH - 70))**2 + (event.pos[1] - 30)**2 <= 20**2:
                    if not playing:
                        playing = True
                        start_time = current_time
                    else:
                        playing = False
                # LOS Dragging
                elif abs(event.pos[0] - los_x) < 5 and 50 < event.pos[1] < HEIGHT - 50: #Check if the click is within the LOS area
                    dragging_los = True
                    los_start_x = event.pos[0]
                else:
                    for circle in circles:
                        dx = event.pos[0] - circle["pos"][0]
                        dy = event.pos[1] - circle["pos"][1]
                        if dx * dx + dy * dy <= CIRCLE_RADIUS * CIRCLE_RADIUS:
                            circle["dragging"] = True
                # handle input box
                if input_box.collidepoint(event.pos):
                    input_box_active = True
                else:
                    input_box_active = False
                if button_rect.collidepoint(event.pos):
                    handle_api_request(input_text, input_box_active)
            elif event.type == pygame.MOUSEBUTTONUP:
                for circle in circles:
                    circle["dragging"] = False
                dragging_los = False # Stop dragging the LOS
            elif event.type == pygame.MOUSEMOTION:
                for circle in circles:
                    if circle["dragging"]:
                        if mode == "Position":
                            new_x, new_y = event.pos
                            # Boundaries
                            if 50 < new_x < WIDTH - 50 and 50 < new_y < HEIGHT - 50:
                                circle["pos"] = [new_x, new_y]
                                # LOS Bounds
                                # if circle["color"] == RED: # and new_x < WIDTH // 2:
                                # elif circle["color"] == BLUE: # and new_x > WIDTH // 2:
                                #     circle["pos"] = [new_x, new_y]
                        elif mode == "Vector":
                            start_pos = circle["pos"]
                            circle["vector"] = [event.pos[0] - start_pos[0], event.pos[1] - start_pos[1]]
                if dragging_los: # move the LOS based on mouse movement
                    los_offset = event.pos[0] - los_start_x
                    los_x += los_offset
                    los_start_x = event.pos[0]
                    update_los(circles, los_offset)
                    # los_x = 
            if event.type == pygame.KEYDOWN:
                if input_box_active:
                    if event.key == pygame.K_RETURN:
                        # Send data to API
                        handle_api_request(input_text, input_box_active)
                    elif event.key == pygame.K_BACKSPACE:
                        input_text = input_text[:-1]
                    else:
                        input_text += event.unicode
                else:
                    # if event.key == pygame.K_SPACE:
                    #     paused = not paused
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
                    elif event.key == pygame.K_0:
                        set_formation("Defense", "Nickel")

        if noise < 1:
            dots = get_noise(0.005)
            noise += 1
        draw_field(dots, los_x)
        draw_toggle(mode)
        draw_play_button(playing)
        draw_play_dropdown()
        draw_input_box(input_box_active, input_text)

        # Get coverage prediction
        zone, man = predict_coverage(circles)
        font = pygame.font.Font(None, 32)
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
        # await asyncio.sleep(0) 

    pygame.quit()
    sys.exit()

main()
