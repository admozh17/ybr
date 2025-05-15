import cv2
import numpy as np
import pytesseract
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import time


class TextRegionDetector:
    """
    Detects text regions in video frames using combined
    blob detection and contrast isolation.
    """

    def __init__(
        self,
        min_area=200,
        max_area=50000,
        min_aspect_ratio=1.5,
        max_aspect_ratio=15,
        min_height=10,
        max_height=200,
        stability_threshold=0.6,
        min_frame_occurrences=2,
    ):
        """
        Initialize with parameters for text region detection.

        Args:
            min_area: Minimum area of a text blob
            max_area: Maximum area of a text blob
            min_aspect_ratio: Minimum width/height ratio for text
            max_aspect_ratio: Maximum width/height ratio for text
            min_height: Minimum height of text in pixels
            max_height: Maximum height of text in pixels
            stability_threshold: How stable a region must be over frames (0-1)
            min_frame_occurrences: Minimum number of frames a region must appear in
        """
        self.min_area = min_area
        self.max_area = max_area
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.min_height = min_height
        self.max_height = max_height
        self.stability_threshold = stability_threshold
        self.min_frame_occurrences = min_frame_occurrences

        # Store detected regions
        self.text_regions = []
        self.region_occurrences = defaultdict(int)
        self.region_frames = {}

    def detect_high_contrast_regions(self, frame):
        """
        Detect high contrast regions that might contain text.

        Args:
            frame: Input video frame

        Returns:
            Binary mask of high-contrast regions
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Use adaptive thresholding to find high-contrast regions
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )

        # Apply morphological operations to connect text areas
        kernel = np.ones((3, 1), np.uint8)  # Horizontal kernel
        dilated_h = cv2.dilate(thresh, kernel, iterations=2)

        kernel = np.ones((1, 3), np.uint8)  # Vertical kernel
        dilated = cv2.dilate(dilated_h, kernel, iterations=1)

        return dilated

    def detect_text_blobs(self, contrast_mask):
        """
        Detect text-like blobs from the high-contrast mask.

        Args:
            contrast_mask: Binary mask of high-contrast regions

        Returns:
            List of potential text regions as (x, y, w, h) rectangles
        """
        # Find connected components (blobs)
        contours, _ = cv2.findContours(
            contrast_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Filter contours by text-like properties
        text_regions = []

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            # Calculate aspect ratio
            aspect_ratio = w / h if h > 0 else 0
            area = w * h

            # Filter by size and aspect ratio
            if (
                self.min_area < area < self.max_area
                and self.min_height < h < self.max_height
                and self.min_aspect_ratio < aspect_ratio < self.max_aspect_ratio
            ):
                text_regions.append((x, y, w, h))

        return text_regions

    def merge_overlapping_regions(self, regions, overlap_threshold=0.5):
        """
        Merge regions that overlap significantly.

        Args:
            regions: List of (x, y, w, h) regions
            overlap_threshold: Minimum IoU for merging

        Returns:
            List of merged regions
        """
        if not regions:
            return []

        # Sort by x coordinate
        sorted_regions = sorted(regions, key=lambda r: r[0])
        merged_regions = [sorted_regions[0]]

        for current in sorted_regions[1:]:
            previous = merged_regions[-1]

            # Calculate coordinates
            curr_x1, curr_y1 = current[0], current[1]
            curr_x2, curr_y2 = current[0] + current[2], current[1] + current[3]

            prev_x1, prev_y1 = previous[0], previous[1]
            prev_x2, prev_y2 = previous[0] + previous[2], previous[1] + previous[3]

            # Check for overlap
            overlap_x1 = max(prev_x1, curr_x1)
            overlap_y1 = max(prev_y1, curr_y1)
            overlap_x2 = min(prev_x2, curr_x2)
            overlap_y2 = min(prev_y2, curr_y2)

            # If there's overlap
            if overlap_x1 < overlap_x2 and overlap_y1 < overlap_y2:
                overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
                curr_area = (curr_x2 - curr_x1) * (curr_y2 - curr_y1)
                prev_area = (prev_x2 - prev_x1) * (prev_y2 - prev_y1)

                # Calculate IoU (Intersection over Union)
                iou = overlap_area / (curr_area + prev_area - overlap_area)

                if iou > overlap_threshold:
                    # Merge the regions
                    merged_x1 = min(prev_x1, curr_x1)
                    merged_y1 = min(prev_y1, curr_y1)
                    merged_x2 = max(prev_x2, curr_x2)
                    merged_y2 = max(prev_y2, curr_y2)

                    merged_w = merged_x2 - merged_x1
                    merged_h = merged_y2 - merged_y1

                    merged_regions[-1] = (merged_x1, merged_y1, merged_w, merged_h)
                else:
                    merged_regions.append(current)
            else:
                merged_regions.append(current)

        return merged_regions

    def analyze_sample_frames(
        self, video_path, num_sample_frames=15, sample_interval=24
    ):
        """
        Analyze a sample of frames to detect consistent text regions.

        Args:
            video_path: Path to video
            num_sample_frames: Number of frames to sample
            sample_interval: Interval between sampled frames
        """
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            print("Error: Could not open video.")
            return False

        # Get video dimensions
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Get FPS and total frames
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Check if we have enough frames
        if total_frames < num_sample_frames * sample_interval:
            sample_interval = max(1, total_frames // num_sample_frames)

        print(f"Analyzing {num_sample_frames} sample frames from video...")

        # Process sample frames
        frame_idx = 0
        frames_processed = 0
        all_regions = []

        while frames_processed < num_sample_frames and frame_idx < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if not ret:
                break

            # Detect high contrast regions
            contrast_mask = self.detect_high_contrast_regions(frame)

            # Detect text blobs
            regions = self.detect_text_blobs(contrast_mask)

            # Merge overlapping regions
            merged_regions = self.merge_overlapping_regions(regions)

            # Track regions
            for region in merged_regions:
                # Create a region key based on position
                # We use a grid-based approach to account for slight movements
                x, y, w, h = region
                region_key = (x // 20, y // 10, w // 20, h // 10)  # Grid cells

                self.region_occurrences[region_key] += 1

                # Store the actual region for this occurrence
                if region_key in self.region_frames:
                    old_x, old_y, old_w, old_h = self.region_frames[region_key]
                    # Average the regions
                    avg_x = (old_x + x) // 2
                    avg_y = (old_y + y) // 2
                    avg_w = (old_w + w) // 2
                    avg_h = (old_h + h) // 2
                    self.region_frames[region_key] = (avg_x, avg_y, avg_w, avg_h)
                else:
                    self.region_frames[region_key] = region

            frames_processed += 1
            frame_idx += sample_interval

        cap.release()

        # Filter for regions that appear in multiple frames
        stable_regions = []
        for region_key, count in self.region_occurrences.items():
            if count >= self.min_frame_occurrences:
                stability_score = count / num_sample_frames

                if stability_score >= self.stability_threshold:
                    stable_regions.append(self.region_frames[region_key])

        # Merge any overlapping stable regions
        self.text_regions = self.merge_overlapping_regions(stable_regions)

        print(f"Found {len(self.text_regions)} stable text regions.")
        return len(self.text_regions) > 0

    def get_text_regions(self):
        """
        Return the detected text regions.

        Returns:
            List of (x, y, w, h) rectangles
        """
        return self.text_regions

    def visualize_text_regions(self, frame):
        """
        Draw the detected text regions on a frame.

        Args:
            frame: Input video frame

        Returns:
            Frame with visualized text regions
        """
        vis_frame = frame.copy()

        for x, y, w, h in self.text_regions:
            cv2.rectangle(vis_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        return vis_frame


def optimized_ocr(video_path, max_frames=30, sample_interval=24, debug=False):
    """
    Process video with optimized OCR focused on detected text regions.
    No character limit - processes all detected frames.

    Args:
        video_path: Path to video file
        max_frames: Maximum number of frames to process for OCR
        sample_interval: Frame interval for OCR processing
        debug: Whether to save debug visualizations

    Returns:
        Extracted text from stable text regions
    """
    video_path = Path(video_path)
    start_time = time.time()

    # Initialize text region detector
    detector = TextRegionDetector()

    # Analyze sample frames to find stable text regions
    print("Phase 1: Detecting stable text regions...")
    found_regions = detector.analyze_sample_frames(video_path)

    if not found_regions:
        print("No stable text regions found. Falling back to bottom-third OCR.")
        return bottom_third_ocr(video_path, max_frames)

    # Get the stable text regions
    text_regions = detector.get_text_regions()

    # Phase 2: Apply OCR only to the detected text regions
    print("Phase 2: Applying OCR to detected text regions...")
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print("Error: Could not open video.")
        return ""

    # Get total frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Process frames for OCR
    frame_idx = 0
    frames_processed = 0
    all_texts = []

    # Create a debug directory if needed
    if debug:
        debug_dir = video_path.parent / "debug_ocr"
        debug_dir.mkdir(exist_ok=True)

    while frames_processed < max_frames and frame_idx < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret:
            break

        # If debug mode, create a visualization
        if debug:
            vis_frame = detector.visualize_text_regions(frame)
            cv2.imwrite(str(debug_dir / f"frame_{frame_idx}.jpg"), vis_frame)

        # Process each text region
        frame_texts = []

        for region_idx, (x, y, w, h) in enumerate(text_regions):
            # Extract region
            region = frame[y : y + h, x : x + w]

            # Apply preprocessing for better OCR
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

            # Apply adaptive thresholding
            thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )

            # Invert if necessary (white text on black background)
            if np.mean(gray) < 127:
                thresh = cv2.bitwise_not(thresh)

            # Save the processed region in debug mode
            if debug:
                cv2.imwrite(
                    str(debug_dir / f"frame_{frame_idx}_region_{region_idx}.jpg"),
                    thresh,
                )

            # Apply OCR
            custom_config = r"--oem 3 --psm 6"
            text = pytesseract.image_to_string(thresh, lang="eng", config=custom_config)

            if text.strip():
                frame_texts.append(text.strip())

        # Combine texts from all regions
        if frame_texts:
            combined_text = " ".join(frame_texts)
            all_texts.append(combined_text)

        frames_processed += 1
        frame_idx += sample_interval

    cap.release()

    # Combine all extracted text
    result = "\n".join(all_texts)

    end_time = time.time()
    print(f"OCR processing completed in {end_time - start_time:.2f} seconds.")
    print(f"Processed {frames_processed} frames.")
    print(f"Extracted {len(result)} characters from {len(all_texts)} frames.")
    output_file = f"{video_path.stem}_ocr_output.txt"
    with open(output_file, "w") as f:
        f.write(result)
    print(f"Raw OCR output saved to {output_file}")
    return result


def bottom_third_ocr(video_path, max_frames=30):
    """
    Fallback OCR that processes only the bottom third of frames.

    Args:
        video_path: Path to video file
        max_frames: Maximum number of frames to process

    Returns:
        Extracted text from the video
    """
    print("Using bottom-third OCR fallback...")
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print("Error: Could not open video.")
        return ""

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate frame step
    step = max(1, frame_count // max_frames)

    texts = []
    frames_processed = 0

    for frame_idx in tqdm(range(0, frame_count, step), desc="OCR"):
        if frames_processed >= max_frames:
            break

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret:
            continue

        # Get dimensions
        height, width = frame.shape[:2]

        # Extract only the bottom third
        bottom_region = frame[height // 3 * 2 : height, :]

        # Process the region
        gray = cv2.cvtColor(bottom_region, cv2.COLOR_BGR2GRAY)

        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        # Invert if necessary
        if np.mean(gray) < 127:
            thresh = cv2.bitwise_not(thresh)

        # Apply OCR
        custom_config = r"--oem 3 --psm 6"
        text = pytesseract.image_to_string(thresh, lang="eng", config=custom_config)

        if text.strip():
            texts.append(text.strip())

        frames_processed += 1

    cap.release()

    result = "\n".join(texts)
    print(
        f"Bottom-third OCR extracted {len(result)} characters from {len(texts)} frames."
    )

    return result
