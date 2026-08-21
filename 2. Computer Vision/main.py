import numpy as np

from ImageProcessor import ImageProcessor

def print_matrix(name, matrix):
    print(f"\n{name}")
    print(matrix)
    print("Shape:", matrix.shape)


def main():
    # Simple image with a strong vertical edge
    image = np.array([
        [10, 10, 10, 200, 200],
        [10, 10, 10, 200, 200],
        [10, 10, 10, 200, 200],
        [10, 10, 10, 200, 200],
        [10, 10, 10, 200, 200],
    ], dtype=float)

    print("Original image:")
    print(image)

    image_processor = ImageProcessor()

    # Gaussian kernel
    kernel = image_processor.gaussian_kernel(
        size=3,
        sigma=1.0
    )

    print("\nGaussian kernel:")
    print(kernel)
    print("Kernel sum:", np.sum(kernel))

    # Blur
    blurred = image_processor.gaussian_blur(
        image,
        size=3,
        sigma=1.0
    )

    gx_raw = image_processor.sobel_x(image)
    gy_raw = image_processor.sobel_y(image)

    mag_raw = image_processor.gradient_magnitude(gx_raw, gy_raw)

    print("Raw Sobel X:")
    print(gx_raw)

    print("Raw Sobel Y:")
    print(gy_raw)

    print("Raw magnitude:")
    print(mag_raw)

    print("\nBlurred image:")
    print(blurred)

    # Sobel gradients
    gx = image_processor.sobel_x(blurred)
    gy = image_processor.sobel_y(blurred)

    print("\nGradient X:")
    print(gx)

    print("\nGradient Y:")
    print(gy)

    # Gradient magnitude
    magnitude = image_processor.gradient_magnitude(
        gx,
        gy
    )

    print("\nGradient magnitude:")
    print(magnitude)

    # Gradient direction
    direction = image_processor.gradient_direction(
        gx,
        gy
    )

    print("\nGradient direction (radians):")
    print(direction)

    print("\nGradient direction (degrees):")
    print(np.degrees(direction))

     # ---------------------------------------------------------
    # Create a simple synthetic image containing a white square.
    # The square has 4 obvious corners.
    #
    # 0   = black
    # 255 = white
    # ---------------------------------------------------------
    image = np.zeros((15, 15), dtype=float)

    image[4:11, 4:11] = 255.0

    print_matrix("Original image:", image)

    # ---------------------------------------------------------
    # Harris response
    # ---------------------------------------------------------
    R = image_processor.harris_response(
        image,
        gaussian_size=3,
        sigma=1.0,
        k=0.04
    )

    print_matrix("Harris response:", R)

    # ---------------------------------------------------------
    # Normalize only for easier viewing.
    #
    # This is NOT required by Harris itself.
    # ---------------------------------------------------------
    max_abs = np.max(np.abs(R))

    if max_abs > 0:
        R_normalized = R / max_abs
    else:
        R_normalized = R

    print_matrix(
        "Normalized Harris response:",
        np.round(R_normalized, 2)
    )

    # ---------------------------------------------------------
    # Keep only strong positive corner responses.
    #
    # Here we choose a relative threshold:
    # keep anything >= 20% of the strongest corner.
    # ---------------------------------------------------------
    threshold = 0.20 * np.max(R)

    corner_mask = R > threshold

    print_matrix(
        "Corner mask:",
        corner_mask.astype(int)
    )

    # ---------------------------------------------------------
    # Get detected coordinates
    # ---------------------------------------------------------
    corner_coordinates = np.argwhere(corner_mask)

    print("\nDetected corner coordinates in R:")
    for y, x in corner_coordinates:
        print(f"(x={x}, y={y})  R={R[y, x]:.2f}")


if __name__ == "__main__":
    main()