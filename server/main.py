from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
import json
from supabase import create_client, Client
from dotenv import load_dotenv
import os
import logging
import torch
from model import model, prepare_tensor
# from utils import predict_coverage, create_player_dataframe
import pandas as pd

logger = logging.getLogger(__name__)

load_dotenv()

# Supabase connection details
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SERVICE_KEY")

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
app = FastAPI()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@app.on_event("startup")
async def load_model():
    """Loads the model once when FastAPI starts."""
    global model
    model.load_state_dict(torch.load('./best_model_week3.pth', weights_only=True, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    print("Model loaded successfully!")

BLUE = (0, 0, 255)

# model.load_state_dict(torch.load('./best_model_week3.pth', weights_only=True, map_location=DEVICE))
# model.eval()


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
  print(positions)
  frame_tensor = prepare_tensor(positions)

  frame_tensor = frame_tensor.to(DEVICE)  # Move to device if necessary

  with torch.no_grad():
      outputs = model(frame_tensor)  # Shape: [num_frames, num_classes]
      probabilities = torch.softmax(outputs, dim=1).cpu().numpy()

      zone_prob = probabilities[0][0]
      man_prob = probabilities[0][1]

  return zone_prob, man_prob

@app.get("/tracking/")
def get_tracking_data(gameId: int, playId: int):
    """
    GET request to retrieve rows from the tracking table based on gameId and playId.

    Query Parameters:
    - gameId: Game ID to filter by.
    - playId: Play ID to filter by.
    """
    try:
        # Query the Supabase table
        response = (
            supabase.table("tracking")
            .select("*")
            .eq("gameId", gameId)
            .eq("playId", playId)
            .eq("frameType", "SNAP")
            .execute()
        )

        # Check for data in the response
        if not response.data:
            raise HTTPException(status_code=404, detail="No data found for the provided gameId and playId.")

        return {"status": "success", "data": response.data}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")


async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for predicting coverage based on player positions."""
    # model.load_state_dict(torch.load('./best_model_week3.pth', weights_only=True, map_location=DEVICE))
    # model.eval()
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            circles = json.loads(data)  # Expecting a list of dictionaries
            logger.info(circles)
            
            zone_prob, man_prob = predict_coverage(circles)
            
            response = {"zone_prob": round(float(zone_prob), 2), "man_prob": round(float(man_prob), 2)}
            await websocket.send_json(response)
    
    except WebSocketDisconnect:
        print("Client disconnected")

@app.websocket("/ws/predict")
async def predict_ws(websocket: WebSocket):
    await websocket_endpoint(websocket)

if __name__ == '__main__':
    print(supabase.table("tracking").select("*").limit(10).execute())