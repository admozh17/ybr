def simple_ocr_fallback(video_path, max_frames=20):
    """
    Simple fallback OCR using basic Tesseract with adaptive thresholding.
    Based on our diagnostic tests showing good results with adaptive thresholding.
    
    Args:
        video_path: Path to video file
        max_frames: Maximum number of frames to process
        
    Returns:
        Extracted text from the video
    """
    import pytesseract
    
    try:
        # Print what we're doing
        print("🔍 Using improved Tesseract OCR with adaptive thresholding...")
        
        # Extract frames using OpenCV
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Sample evenly spaced frames
        interval = max(1, frame_count // max_frames)
        texts = []
        
        for i in range(0, frame_count, interval):
            if len(texts) >= max_frames:
                break
                
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                continue
                
            # Process the FULL frame (not just bottom) based on test results
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (3, 3), 0)
            
            # Apply adaptive thresholding + inversion (our best method from testing)
            thresh = cv2.adaptiveThreshold(
                blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            inverted = cv2.bitwise_not(thresh)
            
            # Run OCR on the inverted version
            text1 = pytesseract.image_to_string(inverted)
            
            # Also try the non-inverted version for comparison
            text2 = pytesseract.image_to_string(thresh)
            
            # Use the longer result
            text = text1 if len(text1) > len(text2) else text2
            
            if text.strip():
                texts.append(text.strip())
                print(f"✓ Frame {i}: Found text ({len(text.strip())} chars)")
                
        result = "\n".join(texts)
        print(f"✅ Improved OCR completed: extracted {len(result)} characters")
        return result
    except Exception as e:
        print(f"Error in fallback OCR: {e}")
        return ""