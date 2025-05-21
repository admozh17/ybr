import cv2
import numpy as np
import os
import sys
import pytesseract
import argparse
import tempfile
import subprocess
from pathlib import Path

def download_video(url, output_path):
    """Download video from URL using yt-dlp"""
    try:
        print(f"📥 Downloading video from {url}...")
        cmd = ["yt-dlp", "-f", "mp4", "-o", str(output_path), url]
        subprocess.run(cmd, check=True)
        print(f"✅ Downloaded video to {output_path}")
        return True
    except Exception as e:
        print(f"❌ Error downloading video: {str(e)}")
        return False

def test_tesseract_installation():
    """Test if Tesseract is properly installed"""
    try:
        version = pytesseract.get_tesseract_version()
        print(f"✅ Tesseract installed: version {version}")
        
        langs = pytesseract.get_languages()
        print(f"✅ Available languages: {', '.join(langs)}")
        return True
    except Exception as e:
        print(f"❌ Tesseract installation issue: {str(e)}")
        return False

def test_ocr_on_image(image_path):
    """Test OCR on a single image with multiple methods"""
    if not os.path.exists(image_path):
        print(f"❌ Image file not found: {image_path}")
        return
    
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Failed to load image: {image_path}")
        return
    
    height, width = image.shape[:2]
    print(f"Image size: {width}x{height} pixels")
    
    # Create output directory for debug images
    debug_dir = "ocr_debug"
    os.makedirs(debug_dir, exist_ok=True)
    
    # Try different OCR approaches and save intermediate images
    approaches = [
        ("original", lambda img: img),
        ("grayscale", lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)),
        ("threshold", lambda img: cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 150, 255, cv2.THRESH_BINARY)[1]),
        ("inverted", lambda img: cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 150, 255, cv2.THRESH_BINARY_INV)[1]),
        ("adaptive", lambda img: cv2.adaptiveThreshold(
            cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (3,3), 0),
            255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)),
        ("adaptive_inv", lambda img: cv2.bitwise_not(cv2.adaptiveThreshold(
            cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (3,3), 0),
            255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2))),
        ("bottom_half", lambda img: img[img.shape[0]//2:, :]),
        ("bottom_third", lambda img: img[int(img.shape[0]*2/3):, :])
    ]
    
    print("\n===== OCR TEST RESULTS =====")
    for name, preprocess in approaches:
        # Process the image
        processed = preprocess(image)
        
        # Save processed image
        output_path = os.path.join(debug_dir, f"{name}.jpg")
        cv2.imwrite(output_path, processed)
        
        # Apply OCR
        config = "--psm 6"
        try:
            text = pytesseract.image_to_string(processed, config=config)
            clean_text = text.strip()
            print(f"\n{name.upper()} APPROACH:")
            print(f"Saved to: {output_path}")
            print(f"Text ({len(clean_text)} chars): {clean_text[:100]}..." if len(clean_text) > 100 else clean_text)
        except Exception as e:
            print(f"❌ Error with {name} approach: {str(e)}")

def test_video_frame_extraction(video_path, num_frames=5):
    """Test frame extraction from video and save sample frames"""
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        return
    
    # Create output directory for frames
    frames_dir = "frame_samples"
    os.makedirs(frames_dir, exist_ok=True)
    
    try:
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"❌ Failed to open video: {video_path}")
            return
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"\n===== VIDEO INFORMATION =====")
        print(f"Resolution: {width}x{height} pixels")
        print(f"FPS: {fps}")
        print(f"Total frames: {frame_count}")
        print(f"Duration: {frame_count/fps:.2f} seconds")
        
        # Extract frames at regular intervals
        interval = max(1, frame_count // num_frames)
        
        for i in range(num_frames):
            frame_idx = i * interval
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                print(f"❌ Failed to read frame at index {frame_idx}")
                continue
            
            # Save frame
            frame_path = os.path.join(frames_dir, f"frame_{frame_idx}.jpg")
            cv2.imwrite(frame_path, frame)
            print(f"Saved frame {frame_idx} to {frame_path}")
            
            # Extract bottom part
            bottom_half = frame[frame.shape[0]//2:, :]
            bottom_path = os.path.join(frames_dir, f"frame_{frame_idx}_bottom.jpg")
            cv2.imwrite(bottom_path, bottom_half)
            
        cap.release()
        print(f"✅ Extracted {num_frames} sample frames")
        
        # If we have frames, try OCR on the first one
        first_frame = os.path.join(frames_dir, f"frame_0.jpg")
        if os.path.exists(first_frame):
            print("\nTesting OCR on first frame:")
            test_ocr_on_image(first_frame)
            
    except Exception as e:
        print(f"❌ Error in video frame extraction: {str(e)}")

def test_easyocr_installation():
    """Test if EasyOCR is properly installed and working"""
    try:
        import easyocr
        print(f"✅ EasyOCR module imported successfully")
        
        reader = easyocr.Reader(['en'], gpu=False)
        print(f"✅ EasyOCR reader initialized")
        
        # Create a simple test image with text
        test_img = np.ones((100, 300, 3), dtype=np.uint8) * 255
        cv2.putText(test_img, "Testing EasyOCR", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Save test image
        test_img_path = "easyocr_test.jpg"
        cv2.imwrite(test_img_path, test_img)
        
        # Try reading the test image
        result = reader.readtext(test_img_path)
        
        if result:
            print(f"✅ EasyOCR successfully read text: {result[0][1]}")
        else:
            print("⚠️ EasyOCR didn't detect any text in the test image")
            
        # Clean up
        os.remove(test_img_path)
        return True
    except ImportError:
        print("❌ EasyOCR is not installed. Install with: pip install easyocr")
        return False
    except Exception as e:
        print(f"❌ EasyOCR error: {str(e)}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OCR Diagnostic Tool")
    parser.add_argument("--video", help="Path to video file for testing")
    parser.add_argument("--url", help="URL to download video from")
    parser.add_argument("--image", help="Path to image file for testing")
    parser.add_argument("--frames", type=int, default=5, help="Number of frames to extract from video")
    args = parser.parse_args()
    
    print("===== OCR DIAGNOSTIC TOOL =====")
    
    # Test Tesseract installation
    tesseract_ok = test_tesseract_installation()
    
    # Test EasyOCR installation 
    easyocr_ok = test_easyocr_installation()
    
    # Handle URL downloads
    if args.url:
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_video = os.path.join(tmp_dir, "downloaded_video.mp4")
            if download_video(args.url, temp_video):
                test_video_frame_extraction(temp_video, args.frames)
            else:
                print("❌ Failed to download or process the video URL")
    # Handle local video file
    elif args.video:
        test_video_frame_extraction(args.video, args.frames)
    # Handle image file
    elif args.image:
        test_ocr_on_image(args.image)
    else:
        print("\nNo video, URL, or image specified for testing.")
        print("Usage examples:")
        print("  python ocr_tester.py --url https://www.youtube.com/watch?v=VIDEOID")
        print("  python ocr_tester.py --video path/to/video.mp4")
        print("  python ocr_tester.py --image path/to/image.jpg")