import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from micromlgen import port

# --- Configuration ---
DATA_FOLDER = 'bci_data'  # The folder where you saved your .csv files
LABELS = {
    # Support the filenames that exist in the workspace and some common variants
    'noise.csv': 0,
    'single.csv': 1,
    'single_blink.csv': 1,
    'double.csv': 2,
    'double_blink.csv': 2,
    'triple.csv': 3,
    'triple_blink.csv': 3,
    # There is a misspelled "quadrple.csv" in the repo; include it and other variants
    'quadrple.csv': 4,
    'quadruple.csv': 4,
    'quadruple_blink.csv': 4
}
# These are our "features". We'll look at 1-second chunks (100 samples)
WINDOW_SIZE = 100 
WINDOW_STEP = 50  # We'll slide the window by 50 samples


# --- 1. Load all data from your .csv files ---
def load_data(folder):
    X = []
    y = []
    print(f"Loading data from {folder}...")
    for filename, label in LABELS.items():
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            print(f"WARNING: File not found: {filepath}")
            continue
        # Read the raw numbers from the file. Be robust to CSV formatting.
        try:
            data = np.loadtxt(filepath, delimiter=',')
        except Exception as e:
            # Try a tolerant fallback parser for lines like "hh:mm:ss.xxx -> 1938"
            print(f"WARNING: Could not load '{filepath}' with np.loadtxt: {e}. Trying fallback parser.")
            import re
            values = []
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as fh:
                    for line in fh:
                        # Find numeric tokens in the line; choose the last one as the signal value
                        nums = re.findall(r'[-+]?\d*\.?\d+', line)
                        if nums:
                            values.append(float(nums[-1]))
            except Exception as e2:
                print(f"WARNING: Fallback parser failed for '{filepath}': {e2}")
                continue

            if len(values) == 0:
                print(f"WARNING: Could not parse numeric values from '{filepath}', skipping.")
                continue

            data = np.array(values, dtype=float)

        # Ensure we have a 1-D array of samples. If the file has multiple columns,
        # use the first column as the signal.
        data = np.atleast_1d(data).astype(float)
        if data.ndim > 1:
            # If it's shape (n, m) pick the first column
            data = data[:, 0]

        # Skip files that are too short for a single window
        if len(data) < WINDOW_SIZE:
            print(f"WARNING: File '{filepath}' has {len(data)} samples < WINDOW_SIZE ({WINDOW_SIZE}), skipping.")
            continue

        # Slide a window across the raw data. Use +1 so the last full window is included.
        for i in range(0, len(data) - WINDOW_SIZE + 1, WINDOW_STEP):
            window = data[i : i + WINDOW_SIZE]
            
            # --- 2. Feature Extraction ---
            # This is where we describe the "shape" of the window
            # to the ML model.
            features = [
                np.max(window),   # The highest peak in the window
                np.mean(window),  # The average value
                np.std(window)    # The "spikiness" of the signal
            ]
            
            X.append(features)
            y.append(label)
            
    print(f"Data loading complete. Found {len(X)} total samples.")
    return np.array(X), np.array(y)

# --- 3. Train the ML Model ---
print("--- Starting ML Training ---")
X, y = load_data(DATA_FOLDER)

if len(X) == 0:
    print("ERROR: No data was loaded. Did you create the `BCI_Data` folder and save your .csv files inside it?")
else:
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # We'll use a Decision Tree. It's fast, simple, and perfect for this.
    model = DecisionTreeClassifier(max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    # Test the model
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"--- Model is trained! ---")
    print(f"Model Accuracy: {acc * 100:.2f}%")

    # --- 4. Convert Model to C++ ---
    print("--- Converting model to C++ (model.h) ---")
    with open('model.h', 'w') as f:
        f.write(port(model, classname='BlinkModel'))

    print("✅ SUCCESS! Your 'model.h' file has been created.")
    print("You can now move to Phase 3.")