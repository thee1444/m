"""
Fall Detection Data Extraction using YOLOv8 Pose
Uses the same YOLOv8 model that MemryX uses, but runs on CPU/GPU
No MemryX hardware required for data extraction
"""

import cv2
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO

class FallDataExtractorYOLOv8:
    def __init__(self, model_path='yolov8m-pose.onnx', sequence_length=30):
        """
        Args:
            model_path: Path to YOLOv8 pose model 
                       Can be: 'yolov8n-pose.pt', 'yolov8s-pose.pt', 
                              'yolov8m-pose.pt', 'yolov8l-pose.pt'
                       Or path to your custom .pt or .onnx file
            sequence_length: Number of frames per sequence (30 = 1 sec at 30fps)
        """
        self.sequence_length = sequence_length
        
        # Load YOLOv8 Pose model
        print(f"Loading YOLOv8 Pose model: {model_path}")
        self.model = YOLO(model_path)
        print("Model loaded successfully!")
        
        # If model_path is just a name (like 'yolov8m-pose.pt'), 
        # Ultralytics will auto-download it
    
    def extract_keypoints_from_video(self, video_path, verbose=False):
        """
        Extract all keypoints from a video
        
        Returns:
            all_keypoints: numpy array of shape (n_frames, 17, 2)
            frame_count: total frames processed
        """
        cap = cv2.VideoCapture(str(video_path))
        all_keypoints = []
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run YOLOv8 pose estimation
            results = self.model(frame, verbose=False)
            
            if len(results) > 0 and results[0].keypoints is not None:
                # Get keypoints from first person detected
                keypoints_data = results[0].keypoints.xy  # Shape: (num_persons, 17, 2)
                
                if len(keypoints_data) > 0:
                    # Take first person (index 0)
                    keypoints = keypoints_data[0].cpu().numpy()  # Shape: (17, 2)
                    all_keypoints.append(keypoints)
                else:
                    # No person detected
                    all_keypoints.append(np.zeros((17, 2)))
            else:
                # No detection
                all_keypoints.append(np.zeros((17, 2)))
            
            frame_count += 1
            
            if verbose and frame_count % 100 == 0:
                print(f"  Processed {frame_count} frames...")
        
        cap.release()
        return np.array(all_keypoints), frame_count
    
    def extract_features(self, keypoints):
        """
        Extract fall detection features from YOLOv8 keypoints
        
        YOLOv8 Pose has 17 keypoints (COCO format):
        0: nose
        1-2: eyes
        3-4: ears
        5-6: shoulders
        7-8: elbows
        9-10: wrists
        11-12: hips
        13-14: knees
        15-16: ankles
        
        Args:
            keypoints: (17, 2) array of keypoint coordinates
        
        Returns:
            Feature vector (8 features)
        """
        if len(keypoints) == 0 or np.all(keypoints == 0):
            return None
        
        # YOLOv8 keypoint indices (COCO format)
        NOSE = 0
        L_SHOULDER, R_SHOULDER = 5, 6
        L_HIP, R_HIP = 11, 12
        L_KNEE, R_KNEE = 13, 14
        L_ANKLE, R_ANKLE = 15, 16
        
        features = []
        
        try:
            # 1. Body center Y (average of shoulders and hips)
            body_parts_y = [keypoints[L_SHOULDER][1], keypoints[R_SHOULDER][1],
                           keypoints[L_HIP][1], keypoints[R_HIP][1]]
            valid_y = [y for y in body_parts_y if y > 0]
            if len(valid_y) > 0:
                body_center_y = np.mean(valid_y)
            else:
                body_center_y = 0.0
            features.append(body_center_y)
            
            # 2. Head Y position
            head_y = keypoints[NOSE][1] if keypoints[NOSE][1] > 0 else 0.0
            features.append(head_y)
            
            # 3. Body height (head to ankles)
            ankle_y_vals = [keypoints[L_ANKLE][1], keypoints[R_ANKLE][1]]
            valid_ankles = [y for y in ankle_y_vals if y > 0]
            if len(valid_ankles) > 0 and head_y > 0:
                ankle_y = np.mean(valid_ankles)
                body_height = ankle_y - head_y
            else:
                body_height = 0.0
            features.append(body_height)
            
            # 4. Body angle (shoulder to hip line angle from vertical)
            shoulder_center = np.mean([keypoints[L_SHOULDER], keypoints[R_SHOULDER]], axis=0)
            hip_center = np.mean([keypoints[L_HIP], keypoints[R_HIP]], axis=0)
            
            if shoulder_center[0] > 0 and hip_center[0] > 0:
                dx = hip_center[0] - shoulder_center[0]
                dy = hip_center[1] - shoulder_center[1]
                body_angle = np.arctan2(dx, dy)
            else:
                body_angle = 0.0
            features.append(body_angle)
            
            # 5. Aspect ratio (bounding box width/height)
            x_coords = keypoints[:, 0]
            y_coords = keypoints[:, 1]
            valid_x = x_coords[x_coords > 0]
            valid_y = y_coords[y_coords > 0]
            
            if len(valid_x) > 0 and len(valid_y) > 0:
                bbox_width = np.max(valid_x) - np.min(valid_x)
                bbox_height = np.max(valid_y) - np.min(valid_y)
                aspect_ratio = bbox_width / (bbox_height + 1e-6)
            else:
                aspect_ratio = 0.0
            features.append(aspect_ratio)
            
            # 6. Hip height (Y coordinate)
            hip_height = hip_center[1] if hip_center[1] > 0 else 0.0
            features.append(hip_height)
            
            # 7. Head-hip distance
            if keypoints[NOSE][0] > 0 and hip_center[0] > 0:
                head_hip_dist = np.linalg.norm(keypoints[NOSE] - hip_center)
            else:
                head_hip_dist = 0.0
            features.append(head_hip_dist)
            
            # 8. Knee angle (approximate using vectors)
            if (keypoints[L_HIP][1] > 0 and keypoints[L_KNEE][1] > 0 and 
                keypoints[L_ANKLE][1] > 0):
                v1 = keypoints[L_HIP] - keypoints[L_KNEE]
                v2 = keypoints[L_ANKLE] - keypoints[L_KNEE]
                dot_product = np.dot(v1, v2)
                norms = np.linalg.norm(v1) * np.linalg.norm(v2)
                if norms > 0:
                    cos_angle = dot_product / norms
                    knee_angle = np.arccos(np.clip(cos_angle, -1, 1))
                else:
                    knee_angle = 0.0
            else:
                knee_angle = 0.0
            features.append(knee_angle)
            
            return np.array(features)
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None
    
    def create_sequences(self, keypoints_list, label):
        """
        Create overlapping sequences from keypoint list
        
        Args:
            keypoints_list: List of keypoints for all frames (n_frames, 17, 2)
            label: 0 for normal, 1 for fall
        
        Returns:
            List of (feature_sequence, label) tuples
        """
        sequences = []
        
        # Create overlapping windows
        stride = self.sequence_length // 2  # 50% overlap
        
        for i in range(0, len(keypoints_list) - self.sequence_length + 1, stride):
            window = keypoints_list[i:i + self.sequence_length]
            
            # Extract features for each frame in window
            feature_sequence = []
            valid_frames = 0
            
            for frame_keypoints in window:
                features = self.extract_features(frame_keypoints)
                if features is not None and not np.all(features == 0):
                    feature_sequence.append(features)
                    valid_frames += 1
                else:
                    # Pad with zeros if no person detected
                    feature_sequence.append(np.zeros(8))
            
            # Only use sequences where most frames have valid detections
            if valid_frames >= self.sequence_length * 0.7:  # At least 70% valid
                feature_sequence = np.array(feature_sequence)
                
                # Add temporal features (velocity - first derivative)
                velocity = np.diff(feature_sequence, axis=0)
                velocity = np.vstack([velocity[0], velocity])  # Pad first frame
                
                # Flatten to single feature vector for XGBoost
                # Shape: (sequence_length * features * 2) for spatial + velocity
                combined = np.hstack([feature_sequence.flatten(), velocity.flatten()])
                sequences.append((combined, label))
        
        return sequences
    
    def process_dataset(self, dataset_root, output_file):
        """
        Process entire Le2i dataset
        
        Expected Le2i structure:
        dataset_root/
            Home/
                Videos/
                    video_001.avi
                    video_002.avi
                    ...
                Annotation_files/
                    video_001.txt
                    video_002.txt (empty if normal, has content if fall)
                    ...
            Coffee_room/
                Videos/
                Annotation_files/
            Office/
            Lecture_room/
        
        Args:
            dataset_root: Path to Le2i dataset root
            output_file: Path to save extracted data (.pkl)
        """
        dataset_root = Path(dataset_root)
        all_sequences = []
        
        # Le2i dataset scenes
        scenes = ['Home', 'Coffee_room', 'Office', 'Lecture_room']
        
        for scene in scenes:
            scene_path = dataset_root / scene
            if not scene_path.exists():
                print(f"Scene '{scene}' not found, skipping...")
                continue
            
            videos_path = scene_path / 'Videos'
            annotations_path = scene_path / 'Annotation_files'
            
            if not videos_path.exists():
                print(f"Videos folder not found in {scene}, skipping...")
                continue
            
            # Get all video files
            video_files = sorted(list(videos_path.glob('*.avi')) + 
                               list(videos_path.glob('*.mp4')))
            
            print(f"\n{'='*60}")
            print(f"Processing {scene} ({len(video_files)} videos)")
            print(f"{'='*60}")
            
            for video_file in tqdm(video_files, desc=f"{scene}"):
                try:
                    # Extract keypoints from video
                    keypoints, frame_count = self.extract_keypoints_from_video(
                        video_file, verbose=False
                    )
                    
                    if len(keypoints) == 0:
                        print(f"\n  ⚠ No keypoints extracted from {video_file.name}")
                        continue
                    
                    # Determine if this video contains a fall
                    label = self.get_label_from_annotation(video_file, annotations_path)
                    
                    # Create sequences
                    sequences = self.create_sequences(keypoints, label)
                    all_sequences.extend(sequences)
                    
                    label_str = "FALL" if label == 1 else "NORMAL"
                    print(f"  ✓ {video_file.name}: {len(sequences)} sequences "
                          f"({frame_count} frames, {label_str})")
                    
                except Exception as e:
                    print(f"\n  ✗ Error processing {video_file.name}: {e}")
                    continue
        
        # Save processed data
        if len(all_sequences) == 0:
            print("\n⚠ WARNING: No sequences were extracted!")
            print("Please check your dataset path and structure.")
            return None, None
        
        X = np.array([seq[0] for seq in all_sequences])
        y = np.array([seq[1] for seq in all_sequences])
        
        print(f"\n{'='*60}")
        print("EXTRACTION SUMMARY")
        print(f"{'='*60}")
        print(f"Total sequences extracted: {len(X)}")
        print(f"  Falls:  {np.sum(y == 1)} ({np.sum(y == 1)/len(y)*100:.1f}%)")
        print(f"  Normal: {np.sum(y == 0)} ({np.sum(y == 0)/len(y)*100:.1f}%)")
        print(f"  Feature dimension: {X.shape[1]}")
        print(f"  Sequence length: {self.sequence_length} frames")
        
        # Save to file
        with open(output_file, 'wb') as f:
            pickle.dump({'X': X, 'y': y, 'sequence_length': self.sequence_length}, f)
        
        print(f"\n✓ Data saved to: {output_file}")
        print("You can now train your XGBoost model!")
        
        return X, y
    
    def get_label_from_annotation(self, video_file, annotations_path):
        """
        Read annotation file to determine if video contains fall
        
        Le2i annotation format:
        - Empty file or no file = normal activity (label 0)
        - File with content (fall frames) = fall (label 1)
        
        Returns:
            1 if fall, 0 if normal
        """
        # Get corresponding annotation file
        video_name = video_file.stem
        annotation_file = annotations_path / f"{video_name}.txt"
        
        # Check if annotation file exists
        if annotation_file.exists():
            try:
                with open(annotation_file, 'r') as f:
                    content = f.read().strip()
                    # If file has content, it's a fall
                    if content:
                        return 1
            except Exception as e:
                print(f"  Warning: Could not read annotation file {annotation_file}: {e}")
        
        # Check filename for 'fall' keyword (some datasets use this)
        if 'fall' in video_name.lower():
            return 1
        
        # Default to normal
        return 0


def main():
    """
    Main function to extract data from Le2i dataset
    """
    # ============ CONFIGURATION ============
    
    # YOLOv8 model - choose one:
    # 'yolov8n-pose.pt' - nano (fastest, least accurate)
    # 'yolov8s-pose.pt' - small
    # 'yolov8m-pose.pt' - medium (recommended - good balance)
    # 'yolov8l-pose.pt' - large (slower, more accurate)
    # Or path to your custom model
    MODEL_PATH = 'yolov8m-pose.pt'
    
    # Dataset path - UPDATE THIS!
    DATASET_ROOT = '/path/to/le2i_dataset'  # <-- CHANGE THIS
    
    # Output file
    OUTPUT_FILE = 'fall_detection_data.pkl'
    
    # Sequence length (frames)
    SEQUENCE_LENGTH = 30  # 30 frames ≈ 1 second at 30fps
    
    # =======================================
    
    print("="*60)
    print("FALL DETECTION DATA EXTRACTION")
    print("Using YOLOv8 Pose (No MemryX Hardware Required)")
    print("="*60)
    
    # Initialize extractor
    extractor = FallDataExtractorYOLOv8(
        model_path=MODEL_PATH,
        sequence_length=SEQUENCE_LENGTH
    )
    
    # Process dataset
    X, y = extractor.process_dataset(DATASET_ROOT, OUTPUT_FILE)
    
    if X is not None:
        print("\n" + "="*60)
        print("✓ EXTRACTION COMPLETE!")
        print("="*60)
        print("\nNext steps:")
        print("1. Load the data:")
        print(f"   data = pickle.load(open('{OUTPUT_FILE}', 'rb'))")
        print("   X, y = data['X'], data['y']")
        print("\n2. Train your XGBoost model")
        print("3. Later, integrate with MemryX hardware")
    else:
        print("\n✗ Extraction failed. Please check your dataset path.")


if __name__ == '__main__':
    main()