import os
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from micromlgen import port

# --- Configuration ---
DATA_FOLDER = 'bci_data'  # The folder where you saved your .csv files
# Map filenames we actually have (be permissive with common variants)
LABELS = {
    'noise.csv': 0,
    'single.csv': 1,
    'single_blink.csv': 1,
    'double.csv': 2,
    'double_blink.csv': 2,
    'triple.csv': 3,
    'triple_blink.csv': 3,
    'quadrple.csv': 4,      # spelling used in this dataset
    'quadruple.csv': 4,
    'quadruple_blink.csv': 4
}
WINDOW_SIZE = 100 
WINDOW_STEP = 50
# We need a threshold to count peaks. Look at your "noise" data
# and pick a number safely above it. (e.g., 2500)
PEAK_THRESHOLD = 2200  # <--- TUNE THIS!

# --- 1. Load all data from your .csv files ---

def count_peaks(window_data):
    """Counts how many times the signal crosses the threshold from below."""
    count = 0
    in_peak = False
    for val in window_data:
        if val > PEAK_THRESHOLD and not in_peak:
            count += 1
            in_peak = True
        elif val < PEAK_THRESHOLD:
            in_peak = False
    return count

def load_data(folder):
    X = []
    y = []
    print(f"Loading data from {folder}...")
    for filename, label in LABELS.items():
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            print(f"WARNING: File not found: {filepath}")
            continue
        # Read the file line-by-line and extract the trailing integer value.
        values = []
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                # Common formats: "<time> -> 1981", "1981", "23:12:40.852,1981"
                # Try to find the last integer on the line.
                m = re.search(r"(-?\d+)(?!.*\d)", line)
                if m:
                    try:
                        values.append(int(m.group(1)))
                    except ValueError:
                        # skip malformed numbers
                        continue
                else:
                    # Attempt a fallback using split and last token
                    parts = re.split(r"[,\s->]+", line)
                    if parts:
                        try:
                            values.append(int(parts[-1]))
                        except Exception:
                            continue

        data = np.array(values, dtype=int)

        if len(data) < WINDOW_SIZE:
            print(f"WARNING: File {filename} has only {len(data)} samples; need at least WINDOW_SIZE={WINDOW_SIZE}. Skipping.")
            continue

        # iterate over all full windows (include the last full window)
        for i in range(0, len(data) - WINDOW_SIZE + 1, WINDOW_STEP):
            window = data[i : i + WINDOW_SIZE]
            
            # --- 2. NEW, SMARTER Feature Extraction ---
            features = [
                np.std(window),    # "Spikiness"
                count_peaks(window) # Our new "killer feature"!
            ]
            
            X.append(features)
            y.append(label)
            
    print(f"Data loading complete. Found {len(X)} total samples.")
    return np.array(X), np.array(y)

# --- 3. Train the ML Model ---
print("--- Starting ML Training (v2) ---")
X, y = load_data(DATA_FOLDER)

if len(X) == 0:
    print(f"ERROR: No data was loaded from '{DATA_FOLDER}'.\n" \
          "Make sure the CSV files are in that folder and that each file contains numeric samples. " \
          "Lines like '23:12:40.852 -> 1981' are supported.")
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = DecisionTreeClassifier(max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"--- Model is trained! ---")
    print(f"Model Accuracy: {acc * 100:.2f}%") # Accuracy should be MUCH higher now

    # --- 4. Convert Model to C++ ---
    print("--- Converting model to C++ (model_v2.h) ---")
    with open('model.h', 'w') as f:
        # NOTE: We MUST change the feature_names to match our new features
        f.write(port(model, classname='BlinkModel', feature_names=['std_dev', 'peak_count']))

    print("✅ SUCCESS! Your NEW 'model.h' file has been created.")