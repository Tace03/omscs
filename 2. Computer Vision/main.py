import numpy as np

from ImageProcessor import ImageProcessor


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


if __name__ == "__main__":
    main()