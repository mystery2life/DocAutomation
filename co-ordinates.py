import sys
import fitz  # PyMuPDF
import cv2
import numpy as np

# ---------------------------------------------------
# Convert PDF → image
# ---------------------------------------------------
def pdf_page_to_image(pdf_path, page_num=0, zoom=2):
    doc = fitz.open(pdf_path)
    page = doc[page_num]

    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)

    img = np.frombuffer(pix.samples, dtype=np.uint8)
    img = img.reshape(pix.h, pix.w, pix.n)

    if img.shape[2] == 4:  # drop alpha if present
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    return img


# ---------------------------------------------------
# Detect largest rectangular contour (the ID card)
# ---------------------------------------------------
def find_largest_quadrilateral(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_quad = None
    best_area = 0

    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(approx) == 4:  # quadrilateral
            area = cv2.contourArea(approx)
            if area > best_area:
                best_area = area
                best_quad = approx

    return best_quad


# ---------------------------------------------------
# Run detection
# ---------------------------------------------------
def get_id_card_coordinates(pdf_path):
    img = pdf_page_to_image(pdf_path)
    quad = find_largest_quadrilateral(img)

    if quad is None:
        print("❌ No rectangular ID-like region detected.")
        return

    pts = quad.reshape(4, 2)

    # Order corners (top-left, top-right, bottom-right, bottom-left)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)

    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]     # top-left
    ordered[1] = pts[np.argmin(diff)]  # top-right
    ordered[2] = pts[np.argmax(s)]     # bottom-right
    ordered[3] = pts[np.argmax(diff)]  # bottom-left

    x, y, w, h = cv2.boundingRect(quad)

    print("\n✅ ID Card Detected!")
    print("----------------------------------")
    print("Corner Points (pixels):")
    print(ordered)
    print("\nBounding Box (x, y, w, h):")
    print((x, y, w, h))
    print("----------------------------------")


# ---------------------------------------------------
# Entry point
# ---------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:  python test.py file.pdf")
        sys.exit(1)

    pdf_path = sys.argv[1]
    get_id_card_coordinates(pdf_path)

