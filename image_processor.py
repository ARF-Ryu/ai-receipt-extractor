import cv2

def preprocess_receipt(image_path: str, output_path: str = "processed_receipt.jpg") -> str:
    """Membersihkan dan mempertajam gambar struk menggunakan OpenCV."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Gambar tidak dapat dibaca oleh OpenCV. Periksa path file.")

    # Konversi ke Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Denoising untuk menghilangkan bintik noise kertas thermal
    denoised = cv2.fastNlMeansDenoising(gray, h=30, templateWindowSize=7, searchWindowSize=21)

    # Adaptive Thresholding untuk ketajaman teks maksimal
    thresh = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    cv2.imwrite(output_path, thresh)
    return output_path