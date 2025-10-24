"""
Fall Detection Data Extraction Pipeline
Extracts pose keypoints from Le2i dataset videos using MemryX
Prepares data for XGBoost training
"""

import cv2
import numpy as np
import json
import os
from pathlib import Path
from memryx import AsyncAccl
import pickle
from tqdm import tqdm

class FallDataExtractor:
    def __init__(self, model_path, sequence_length=30):
        """
        Args:
            model_path: Path to MemryX pose model (.dfp file)
            sequence_length: Number of frames per sequence (30 = 1 sec at 30fps)
        """
        self.sequence_length = sequence_length
        self.accl = AsyncAccl(model_path)
        
    def extract_keypoints_from_video(self, video_path):
        """Extract all keypoints from a video"""
        cap = cv2.VideoCapture(str(video_path))
        all_keypoints = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run pose estimation
            outputs = self.accl.run(frame)
            
            # Extract keypoints from first person detected
            if outputs and len(outputs) > 0:
                # Assuming outputs[0] contains keypoints as (17, 2) array
                keypoints = outputs[0].get('keypoints', None)
                if keypoints is not None and len(keypoints) == 17:
                    all_keypoints.append(keypoints)
            else:
                # No person detected, use zeros
                all_keypoints.append(np.zeros((17, 2)))
        
        cap.release()
        return np.array(all_keypoints)
    
    def extract_features(self, keypoints):
        """
        Extract fall detection features from keypoints
        
        Args:
            keypoints: (17, 2) array of keypoint coordinates
        
        Returns:
            Feature vector
        """
        if len(keypoints) == 0 or np.all(keypoints == 0):
            return None
        
        # Key indices
        NOSE = 0
        L_SHOULDER, R_SHOULDER = 5, 6
        L_HIP, R_HIP = 11, 12
        L_KNEE, R_KNEE = 13, 14
        L_ANKLE, R_ANKLE = 15, 16
        
        features = []
        
        # 1. Body center Y (average of shoulders and hips)
        body_parts_y = [keypoints[L_SHOULDER][1], keypoints[R_SHOULDER][1],
                       keypoints[L_HIP][1], keypoints[R_HIP][1]]
        body_center_y = np.mean([y for y in body_parts_y if y > 0])
        features.append(body_center_y)
        
        # 2. Head Y position
        head_y = keypoints[NOSE][1]
        features.append(head_y)
        
        # 3. Body height (head to ankles)
        ankle_y = np.mean([keypoints[L_ANKLE][1], keypoints[R_ANKLE][1]])
        body_height = ankle_y - head_y
        features.append(body_height)
        
        # 4. Body angle (shoulder to hip line angle from vertical)
        shoulder_center = np.mean([keypoints[L_SHOULDER], keypoints[R_SHOULDER]], axis=0)
        hip_center = np.mean([keypoints[L_HIP], keypoints[R_HIP]], axis=0)
        dx = hip_center[0] - shoulder_center[0]
        dy = hip_center[1] - shoulder_center[1]
        body_angle = np.arctan2(dx, dy)
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
            features.append(aspect_ratio)
        else:
            features.append(0.0)
        
        # 6. Hip height (distance from bottom of frame)
        hip_height = hip_center[1]
        features.append(hip_height)
        
        # 7. Head-hip distance
        head_hip_dist = np.linalg.norm(keypoints[NOSE] - hip_center)
        features.append(head_hip_dist)
        
        # 8. Knee angle (approximate)
        if keypoints[L_HIP][1] > 0 and keypoints[L_KNEE][1] > 0 and keypoints[L_ANKLE][1] > 0:
            v1 = keypoints[L_HIP] - keypoints[L_KNEE]
            v2 = keypoints[L_ANKLE] - keypoints[L_KNEE]
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            knee_angle = np.arccos(np.clip(cos_angle, -1, 1))
            features.append(knee_angle)
        else:
            features.append(0.0)
        
        return np.array(features)
    
    def create_sequences(self, keypoints_list, label):
        """
        Create overlapping sequences from keypoint list
        
        Args:
            keypoints_list: List of keypoints for all frames
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
                if features is not None:
                    feature_sequence.append(features)
                    valid_frames += 1
                else:
                    # Pad with zeros if no person detected
                    feature_sequence.append(np.zeros(8))
            
            # Only use sequences where most frames have valid detections
            if valid_frames >= self.sequence_length * 0.7:
                feature_sequence = np.array(feature_sequence)
                
                # Add temporal features (velocity)
                velocity = np.diff(feature_sequence, axis=0)
                velocity = np.vstack([velocity[0], velocity])  # Pad first frame
                
                # Flatten to single feature vector
                combined = np.hstack([feature_sequence.flatten(), velocity.flatten()])
                sequences.append((combined, label))
        
        return sequences
    
    def process_dataset(self, dataset_root, output_file):
        """
        Process entire Le2i dataset
        
        Expected structure:
        dataset_root/
            Home/
                Videos/
                    video-001.avi (fall or normal)
                Annotation_files/
                    annotation-001.txt
            Coffee_room/
                Videos/
                Annotation_files/
            ...
        
        Args:
            dataset_root: Path to Le2i dataset root
            output_file: Path to save extracted data (.pkl)
        """
        dataset_root = Path(dataset_root)
        all_sequences = []
        
        # Process each scene
        scenes = ['Home', 'Coffee_room', 'Office', 'Lecture_room']
        
        for scene in scenes:
            scene_path = dataset_root / scene
            if not scene_path.exists():
                print(f"Scene {scene} not found, skipping...")
                continue
            
            videos_path = scene_path / 'Videos'
            annotations_path = scene_path / 'Annotation_files'
            
            if not videos_path.exists():
                continue
            
            video_files = sorted(videos_path.glob('*.avi'))
            
            print(f"\nProcessing {scene} ({len(video_files)} videos)...")
            
            for video_file in tqdm(video_files):
                # Extract keypoints from video
                keypoints = self.extract_keypoints_from_video(video_file)
                
                if len(keypoints) == 0:
                    print(f"No keypoints extracted from {video_file.name}")
                    continue
                
                # Determine if this video contains a fall
                label = self.get_label_from_annotation(video_file, annotations_path)
                
                # Create sequences
                sequences = self.create_sequences(keypoints, label)
                all_sequences.extend(sequences)
                
                print(f"  {video_file.name}: {len(sequences)} sequences (label={label})")
        
        # Save processed data
        X = np.array([seq[0] for seq in all_sequences])
        y = np.array([seq[1] for seq in all_sequences])
        
        print(f"\nTotal sequences extracted: {len(X)}")
        print(f"  Falls: {np.sum(y == 1)}")
        print(f"  Normal: {np.sum(y == 0)}")
        print(f"  Feature dimension: {X.shape[1]}")
        
        with open(output_file, 'wb') as f:
            pickle.dump({'X': X, 'y': y}, f)
        
        print(f"\nData saved to {output_file}")
        return X, y
    
    def get_label_from_annotation(self, video_file, annotations_path):
        """
        Read annotation file to determine if video contains fall
        
        Annotation format (if exists):
        - start_frame end_frame (indicates fall period)
        
        Returns:
            1 if fall, 0 if normal
        """
        video_name = video_file.stem
        annotation_file = annotations_path / f"{video_name}.txt"
        
        # Check if annotation file exists
        if annotation_file.exists():
            with open(annotation_file, 'r') as f:
                content = f.read().strip()
                if content:  # Non-empty means fall is annotated
                    return 1
        
        # If no annotation or empty, check filename
        # Many datasets use naming conventions
        if 'fall' in video_name.lower():
            return 1
        
        return 0


def main():
    # Configuration
    MEMRYX_MODEL_PATH = 'path/to/yolov8_pose.dfp'  # Update this
    DATASET_ROOT = 'path/to/le2i_dataset'  # Update this
    OUTPUT_FILE = 'fall_detection_data.pkl'
    
    # Initialize extractor
    extractor = FallDataExtractor(
        model_path=MEMRYX_MODEL_PATH,
        sequence_length=30  # 30 frames = ~1 second
    )
    
    # Process dataset
    X, y = extractor.process_dataset(DATASET_ROOT, OUTPUT_FILE)
    
    print("\nData extraction complete!")
    print("You can now train your XGBoost model using this data.")
    print(f"Load with: data = pickle.load(open('{OUTPUT_FILE}', 'rb'))")


if __name__ == '__main__':
    main()