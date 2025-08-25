# test_ugm.py

from obspy.clients.fdsn import Client
from obspy import UTCDateTime

# --- Configuration ---
CLIENT_URL = "geofon" # Obspy can use a simple name for known data centers
NETWORK = "GE"
STATION = "BBJI"
CHANNELS = "BH?" # Broadband High-gain channels (BHZ, BHN, BHE)

# --- Define Time Window ---
# Get the last 10 minutes of data
endtime = UTCDateTime.now()
starttime = endtime - 10 * 60  # 10 minutes ago

print(f"--- Testing Data Availability for {NETWORK}.{STATION} ---")
print(f"Requesting data from: {starttime.isoformat()}")
print(f"To:                   {endtime.isoformat()}")
print("-------------------------------------------------")

try:
    # --- Connect to the data center and request waveforms ---
    client = Client(CLIENT_URL)
    st = client.get_waveforms(
        network=NETWORK,
        station=STATION,
        location="*",
        channel=CHANNELS,
        starttime=starttime,
        endtime=endtime
    )

    # --- Check the result ---
    if st:
        print("\n✅ SUCCESS: Data received!")
        print("Here is a summary of the data stream:")
        print(st)
    else:
        print("\n⚠️ NO DATA: The request was successful, but no data was available for this time window.")
        print("This confirms the station is likely online but currently quiet.")

except Exception as e:
    print(f"\n❌ ERROR: An error occurred while trying to fetch data.")
    print(f"Details: {e}")