from ultralytics import YOLO
import cv2
import numpy as np
import torch
import torch.nn as nn

pose_model = YOLO('yolov8n-pose.pt')  # small pose model, faster

def extract_pose(video_path):
    cap = cv2.VideoCapture(video_path)
    poses = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        results = pose_model.predict(frame, verbose=False)

        if results and results[0].keypoints is not None:
            # Get all detected people
            kpts = results[0].keypoints.xy.cpu().numpy()  # (num_people, 17, 2) ideally

            print(f"[DEBUG] Frame {frame_idx}: Detected {kpts.shape[0]} people shape {kpts.shape}")

            if kpts.shape[0] > 0 and kpts.shape[1] == 17:
                print(f"[DEBUG] Frame {frame_idx}: Valid detection. Taking first person.")
                keypoints = kpts[0]  # (17,2)
                keypoints = keypoints.flatten()
            else:
                print(f"[WARNING] Frame {frame_idx}: No valid person detected or wrong shape {kpts.shape}. Filling zeros.")
                keypoints = np.zeros(34)
        else:
            print(f"[WARNING] Frame {frame_idx}: No detection result. Filling zeros.")
            keypoints = np.zeros(34)

        poses.append(keypoints)

    cap.release()

    poses_array = np.stack(poses)
    print(f"[INFO] Pose extraction complete. Total frames processed: {poses_array.shape[0]}")
    print(f"[INFO] Pose array shape: {poses_array.shape}")
    return poses_array


def normalize_pose(pose):
    pose = pose.reshape(17, 2)

    # Use hips (11,12) and shoulders (5,6) as anchors
    keypoints_used = [11, 12, 5, 6]
    valid = [i for i in keypoints_used if not np.all(pose[i] == 0)]

    if len(valid) >= 2:
        center = np.mean([pose[i] for i in valid], axis=0)
        pose -= center

        # Scale based on torso length
        if len(valid) == 4:
            torso_size = np.linalg.norm(pose[5] - pose[11]) + np.linalg.norm(pose[6] - pose[12])
            if torso_size > 1e-5:  # Only scale if torso_size is large enough
                pose /= (torso_size / 2.0)
            else:
                # If torso too small, fallback to just centering without scaling
                pass
    else:
        pose -= np.mean(pose, axis=0)  # fallback center without scaling

    return pose.flatten()


def make_windows(pose_array, window_size=30, stride=10):
    X = []
    for start in range(0, len(pose_array) - window_size + 1, stride):
        window = pose_array[start:start + window_size]
        window = np.array([normalize_pose(frame) for frame in window])
        X.append(window)
    return np.array(X)  # shape: (num_windows, 30, 34)

path = '/content/drive/MyDrive/training_data/'

punch_pose_array = extract_pose(path +'punching.mp4')
kick_pose_array = extract_pose(path + 'kicking.mp4')
standing_pose_array = extract_pose(path + 'standing.mp4')

punch_windows = make_windows(punch_pose_array)
kick_windows = make_windows(kick_pose_array)
stand_windows = make_windows(standing_pose_array)

X = np.concatenate([punch_windows, kick_windows, stand_windows], axis=0)
y = np.concatenate([
    np.ones(len(punch_windows)),
    2*np.ones(len(kick_windows)),
    np.zeros(len(stand_windows))
], axis=0)

# Shuffle
indices = np.random.permutation(len(X))
X = X[indices]
y = y[indices]

# Check for bad windows
is_finite = np.isfinite(X).all(axis=(1,2))  # (batch dimension)

# Filter out bad samples
X = X[is_finite]
y = y[is_finite]

print("Filtered X shape:", X.shape)
print("Filtered y shape:", y.shape)

class TCN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(TCN, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = x.transpose(1, 2)  # (batch, features, sequence)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.pool(x).squeeze(-1)
        x = self.fc(x)
        return x

import torch
from torch.utils.data import Dataset, DataLoader

class PoseSequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)  # (N, 30, 34)
        self.y = torch.tensor(y, dtype=torch.long)     # (N,)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

dataset = PoseSequenceDataset(X, y)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

model = TCN(input_dim=34, num_classes=3)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(30):
    model.train()
    total_loss = 0
    for batch_x, batch_y in dataloader:
        optimizer.zero_grad()
        preds = model(batch_x)
        loss = loss_fn(preds, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch}: Loss = {total_loss / len(dataloader):.4f}")

# Save model after training
torch.save(model.state_dict(), 'tcn_action_model.pth')

tcn_model = TCN(input_dim=34, num_classes=3)
tcn_model.load_state_dict(torch.load('tcn_action_model.pth'))
tcn_model.eval()

# --- Label mapping ---
label_map = {0: "stand", 1: "punch", 2: "kick"}

# --- Live pose window ---
pose_window = []

# --- Open input video ---
cap = cv2.VideoCapture('MikeTyson.mp4')

# --- Prepare output video writer ---
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_labeled.mp4', fourcc, 30.0,
                      (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                       int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1

    # --- Run pose detection ---
    results = pose_model.predict(frame, verbose=False)

    if results and results[0].keypoints is not None:
        kpts = results[0].keypoints.xy.cpu().numpy()  # shape (num_people, 17, 2)

        if kpts.shape[0] > 0 and kpts.shape[1] == 17:
            # --- Loop through each detected person ---
            for person_idx in range(kpts.shape[0]):
                keypoints = kpts[person_idx].flatten()

                # Normalize pose
                keypoints = normalize_pose(keypoints)

                # Update a separate pose window for each person if you want longer tracking
                # Here we simply predict frame-by-frame

                # Prepare input
                input_seq = torch.tensor([[keypoints] * 30], dtype=torch.float32)  # fake a 30-frame window by repeating
                with torch.no_grad():
                    pred = tcn_model(input_seq)
                    pred_class = pred.argmax(dim=1).item()
                    label = label_map[pred_class]

                # --- Draw bounding box for each person ---
                if results[0].boxes is not None and len(results[0].boxes.xyxy) > person_idx:
                    bbox = results[0].boxes.xyxy.cpu().numpy()[person_idx]
                    x1, y1, x2, y2 = bbox.astype(int)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            pass  # No valid keypoints detected
    else:
        pass  # No results

    # --- Save frame to output video ---
    out.write(frame)

cap.release()
out.release()
print("✅ Finished saving output_labeled.mp4")
