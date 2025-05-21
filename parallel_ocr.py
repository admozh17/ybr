import cv2
import numpy as np
import pytesseract
from pathlib import Path
import concurrent.futures
from tqdm import tqdm
import time
from collections import defaultdict


def extract_frames(video_path, max_frames=30, sample_interval=None):
    """
    Extract frames from video at regular intervals.

    Args:
        video_path: Path to the video file
        max_frames: Maximum number of frames to extract
        sample_interval: Interval between frames (if None, calculated automatically)

    Returns:
        List of (frame_index, frame) tuples
    """
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print("Error: Could not open video.")
        return []

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate sampling interval if not provided
    if sample_interval is None:
        sample_interval = max(1, frame_count // max_frames)

    # Extract frames
    frames = []
    for frame_idx in range(0, frame_count, sample_interval):
        if len(frames) >= max_frames:
            break

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret:
            continue

        frames.append((frame_idx, frame))

    cap.release()
    return frames


def detect_text_regions(frame):
    """
    Detect regions in a frame that likely contain text.

    Args:
        frame: Input video frame

    Returns:
        List of (x, y, w, h) regions that likely contain text
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    # Apply morphological operations to connect text areas
    kernel = np.ones((3, 1), np.uint8)  # Horizontal kernel
    dilated_h = cv2.dilate(thresh, kernel, iterations=2)

    kernel = np.ones((1, 3), np.uint8)  # Vertical kernel
    dilated = cv2.dilate(dilated_h, kernel, iterations=1)

    # Find connected components (blobs)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours by text-like properties
    text_regions = []
    min_area = 200
    max_area = 50000
    min_height = 10
    max_height = 200
    min_aspect_ratio = 1.5
    max_aspect_ratio = 15

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # Calculate aspect ratio
        aspect_ratio = w / h if h > 0 else 0
        area = w * h

        # Filter by size and aspect ratio
        if (
            min_area < area < max_area
            and min_height < h < max_height
            and min_aspect_ratio < aspect_ratio < max_aspect_ratio
        ):
            text_regions.append((x, y, w, h))

    return text_regions


def process_frame_batch(batch):
    """
    Process a batch of frames to extract text.

    Args:
        batch: List of (frame_idx, frame, regions) tuples

    Returns:
        List of (frame_idx, text) tuples
    """
    results = []

    for frame_idx, frame, regions in batch:
        frame_texts = []

        # If no regions provided, process the whole frame
        if not regions:
            # Process the entire frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )

            # Invert if necessary (white text on black background)
            if np.mean(gray) < 127:
                thresh = cv2.bitwise_not(thresh)

            # Apply OCR
            custom_config = r"--oem 3 --psm 6"
            text = pytesseract.image_to_string(thresh, lang="eng", config=custom_config)

            if text.strip():
                frame_texts.append(text.strip())
        else:
            # Process each text region
            for x, y, w, h in regions:
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

                # Apply OCR
                custom_config = r"--oem 3 --psm 6"
                text = pytesseract.image_to_string(
                    thresh, lang="eng", config=custom_config
                )

                if text.strip():
                    frame_texts.append(text.strip())

        # Combine texts from all regions
        combined_text = " ".join(frame_texts) if frame_texts else ""
        results.append((frame_idx, combined_text))

    return results


def parallel_ocr(video_path, max_frames=30, batch_size=5, num_workers=None):
    """
    Process video with parallel OCR.

    Args:
        video_path: Path to video file
        max_frames: Maximum number of frames to process
        batch_size: Number of frames to process in each batch
        num_workers: Number of parallel workers (if None, uses CPU count)

    Returns:
        Extracted text from the video
    """
    start_time = time.time()
    print(
        f"🔍 Starting parallel OCR with max_frames={max_frames}, batch_size={batch_size}"
    )

    video_path = Path(video_path)

    # Determine number of workers based on CPU count if not specified
    if num_workers is None:
        import multiprocessing

        num_workers = max(1, multiprocessing.cpu_count() - 1)

    # Extract frames from video
    print("📹 Extracting frames from video...")
    frames = extract_frames(video_path, max_frames)
    print(f"✅ Extracted {len(frames)} frames for processing")

    # Detect text regions in each frame
    print("🔍 Detecting text regions in frames...")
    frames_with_regions = []
    for frame_idx, frame in frames:
        regions = detect_text_regions(frame)
        frames_with_regions.append((frame_idx, frame, regions))

    # Group frames into batches
    batches = [
        frames_with_regions[i : i + batch_size]
        for i in range(0, len(frames_with_regions), batch_size)
    ]
    print(f"📦 Created {len(batches)} batches of frames for parallel processing")

    # Process batches in parallel
    results = []
    print(f"🚀 Processing batches with {num_workers} workers...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Map batches to process_frame_batch function
        futures = list(
            tqdm(
                executor.map(process_frame_batch, batches),
                total=len(batches),
                desc="OCR Processing",
            )
        )

        # Combine results from all batches
        for batch_results in futures:
            results.extend(batch_results)

    # Sort results by frame index
    results.sort(key=lambda x: x[0])

    # Combine text from all frames
    all_texts = [text for _, text in results if text]
    result = "\n".join(all_texts)

    end_time = time.time()
    print(f"✅ Parallel OCR completed in {end_time - start_time:.2f} seconds")
    print(f"   Processed {len(frames)} frames using {num_workers} workers")
    print(
        f"   Extracted {len(result)} characters from {len(all_texts)} frames with text"
    )

    return result


# Drop-in replacement for the existing ocr_frames function
def parallel_ocr_frames(video_path, max_frames=30):
    """
    Drop-in replacement for ocr_frames function that uses parallel processing.
    """
    return parallel_ocr(video_path, max_frames=max_frames)


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        text = parallel_ocr(video_path)
        print(f"Extracted text ({len(text)} chars):")
        print(text[:500] + "..." if len(text) > 500 else text)
    else:
        print("Usage: python parallel_ocr.py video_path")