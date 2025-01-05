from fastapi import FastAPI, HTTPException
from supabase import create_client, Client
from dotenv import load_dotenv
import os
import logging

logger = logging.getLogger(__name__)

load_dotenv()

# Supabase connection details
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SERVICE_KEY")

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
app = FastAPI()

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

if __name__ == '__main__':
    print(supabase.table("tracking").select("*").limit(10).execute())