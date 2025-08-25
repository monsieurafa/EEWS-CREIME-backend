# inspect_model.py
import tensorflow as tf

SAVED_MODEL_PATH = "creime_savedmodel"

print(f"--- Inspecting Model at: {SAVED_MODEL_PATH} ---")

# Load the low-level model object
model = tf.saved_model.load(SAVED_MODEL_PATH)

# Get the default signature, which is what TFSMLayer uses
signature = model.signatures['serving_default']

print("\n--- Model Output Names ---")
# This will print the exact names of all the outputs
print(list(signature.structured_outputs.keys()))
print("--------------------------")