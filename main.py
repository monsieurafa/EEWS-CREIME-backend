import asyncio
import json
from contextlib import asynccontextmanager
from typing import Dict, List

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from obspy import Trace
from obspy.clients.seedlink.easyseedlink import create_client
from scipy.signal import detrend, butter, filtfilt
from tensorflow.keras.layers import TFSMLayer

# --- 1. Configuration & Global State ---
TARGET_SAMPLING_RATE = 100.0
MODEL_INPUT_SAMPLES = 512
SAVED_MODEL_PATH = "creime_savedmodel"

# Global variables to hold the model, event loop, and data buffers
ml_model = None
main_event_loop = None
station_buffers: Dict[str, Dict[str, np.ndarray]] = {}

# --- 2. Preprocessing & Post-processing Utilities ---
def detrend_trace(trace_data):
    """Applies linear detrending to each channel of the waveform data."""
    detrended_trace = np.zeros_like(trace_data)
    for i in range(trace_data.shape[1]):
        detrended_trace[:, i] = detrend(trace_data[:, i], type='linear')
    return detrended_trace

def bandpass_filter_trace(trace_data, lowcut=1.0, highcut=40.0, fs=100.0, order=4):
    """Applies a bandpass filter to each channel of the waveform data."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    filtered_trace = np.zeros_like(trace_data)
    for i in range(trace_data.shape[1]):
        filtered_trace[:, i] = filtfilt(b, a, trace_data[:, i])
    return filtered_trace

def postprocess_prediction(raw_prediction):
    """Converts the model's raw output into a human-readable prediction tuple."""
    y_pred = raw_prediction[0]
    p_arrival = 512 - np.sum((y_pred > -0.5).astype(int))
    magnitude_proxy = np.mean(y_pred[-10:])
    is_event = not (magnitude_proxy < -0.5)

    if not is_event:
        return (0, None, None)
    else:
        return (1, float(round(magnitude_proxy, 1)), int(p_arrival))

# --- 3. WebSocket Connection Manager ---
class ConnectionManager:
    """Manages active WebSocket connections for broadcasting."""
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    async def connect(self, ws):
        await ws.accept()
        self.active_connections.append(ws)
    def disconnect(self, ws):
        self.active_connections.remove(ws)
    async def broadcast(self, msg):
        for conn in self.active_connections:
            await conn.send_text(msg)
manager = ConnectionManager()

# --- 4. Real-time Data Handling ---
def process_data(trace: Trace):
    """Callback function executed by ObsPy for each new data packet."""
    global main_event_loop
    # Ensure the data matches the model's expected sampling rate
    if trace.stats.sampling_rate != TARGET_SAMPLING_RATE:
        trace.resample(TARGET_SAMPLING_RATE)

    station_id = f"{trace.stats.network}.{trace.stats.station}"
    channel = trace.stats.channel

    # Initialize buffers for new stations/channels
    if station_id not in station_buffers: station_buffers[station_id] = {}
    if channel not in station_buffers[station_id]: station_buffers[station_id][channel] = np.array([])
    
    # Append new data to the corresponding buffer
    station_buffers[station_id][channel] = np.concatenate([station_buffers[station_id][channel], trace.data])
    
    active_channels = station_buffers[station_id]
    
    # Check if we have 3 channels and all have enough data for a prediction
    if len(active_channels) == 3 and all(len(buf) >= MODEL_INPUT_SAMPLES for buf in active_channels.values()):
        
        # Extract the most recent window of data from each channel
        window_bhe = active_channels['BHE'][-MODEL_INPUT_SAMPLES:]
        window_bhn = active_channels['BHN'][-MODEL_INPUT_SAMPLES:]
        window_bhz = active_channels['BHZ'][-MODEL_INPUT_SAMPLES:]
        
        # Preprocess the data window
        waveform_window = np.stack([window_bhe, window_bhn, window_bhz], axis=-1)
        detrended = detrend_trace(waveform_window)
        filtered = bandpass_filter_trace(detrended)
        model_input = np.expand_dims(filtered, axis=0).astype(np.float32)
        
        # Run prediction and post-process the result
        prediction_dict = ml_model(model_input)
        raw_prediction = prediction_dict['dense'].numpy()
        pred_tuple = postprocess_prediction(raw_prediction)

        # Create the payload to send to the frontend
        payload = {
            "type": "new_data_window",
            "data": {
                "ch1": window_bhe.tolist(),
                "ch2": window_bhn.tolist(),
                "ch3": window_bhz.tolist(),
                "prediction": pred_tuple,
                "station_id": station_id
            }
        }
        
        # Safely broadcast the message from the synchronous obspy thread to the async server
        if main_event_loop:
            asyncio.run_coroutine_threadsafe(manager.broadcast(json.dumps(payload)), main_event_loop)
        
        # Trim buffers to manage memory
        for ch in active_channels:
            station_buffers[station_id][ch] = station_buffers[station_id][ch][-MODEL_INPUT_SAMPLES:]

def run_obspy_client():
    """Initializes and runs the ObsPy client in a blocking loop."""
    try:
        client = create_client('geofon.gfz-potsdam.de:18000', on_data=process_data)
        client.select_stream('GE', 'PLAI', 'BH?')
        client.run()
    except Exception as e:
        print(f"--- FATAL ERROR in ObsPy client thread: {e} ---")

# --- 5. FastAPI Application Setup ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles startup and shutdown events for the server."""
    global ml_model, main_event_loop
    main_event_loop = asyncio.get_running_loop() # Get the main event loop
    
    print("INFO:     Loading TensorFlow SavedModel...")
    ml_model = TFSMLayer(SAVED_MODEL_PATH, call_endpoint='serving_default')
    print("INFO:     Model loaded successfully.")
    
    print("INFO:     Starting ObsPy client in background...")
    main_event_loop.run_in_executor(None, run_obspy_client)
    yield
    print("INFO:     Server shutting down.")

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:3000"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: int):
    """The main WebSocket endpoint for frontend clients."""
    await manager.connect(websocket)
    print(f"INFO:     Client #{client_id} connected.")
    try:
        while True:
            await websocket.receive_text() # Keep connection alive
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        print(f"INFO:     Client #{client_id} disconnected.")