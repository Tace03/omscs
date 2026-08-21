import numpy as np

class ImageProcessor:
    def __init__(self):
        return

    def cross_correlate(self, image, kernel):
        image_shape = image.shape 
        kernel_shape = kernel.shape
        output = np.zeros((
            image_shape[0] - kernel_shape[0] + 1,
            image_shape[1] - kernel_shape[1] + 1
        ))

        output_shape = output.shape

        print(f"Output shape is {output_shape}")

        for row in range(output_shape[0]):
            for col in range(output_shape[1]):
                patch = image[
                    row: row + kernel_shape[0], 
                    col: col + kernel_shape[1]
                ]

                multiplied = np.multiply(patch, kernel)
                output[row, col] = np.sum(multiplied)

        return output
    
    def gaussian_kernel(self, size, sigma):
        if (size % 2 == 0):
            return None
        
        output_kernel = np.zeros((size, size))
        sum = 0

        center_row = size // 2
        center_col = size // 2

        for row in range(0,size):
            for col in range(0,size):
                output_kernel[row , col] = np.exp( 
                    -((row - center_row)**2 + (col - center_col)**2)
                    / (2 * sigma ** 2)
                )

                sum += output_kernel[row, col]

        # standardize so the sum is 1
        output_kernel = output_kernel / sum

        return output_kernel

    
    def gaussian_blur(self, image,size = 3,sigma = 2):
        kernel = self.gaussian_kernel(size, sigma)

        return self.cross_correlate(image, kernel)

    def sobel_x(self, image):
        kernel = np.array([
            [-1.0, 0.0, 1.0],
            [-2.0, 0.0, 2.0],
            [-1.0, 0.0, 1.0],
        ])

        return self.cross_correlate(image, kernel)

    def sobel_y(self, image):
        kernel = np.array([
            [-1.0, -2.0, -1.0],
            [ 0.0,  0.0,  0.0],
            [ 1.0,  2.0,  1.0]
        ])

        return self.cross_correlate(image, kernel)

    def gradient_magnitude(self, gx, gy):
        gx = gx ** 2
        gy = gy ** 2

        return np.sqrt(gx + gy)

    def gradient_direction(self, gx, gy):
        return np.arctan2(gy, gx)

    # Return a matrix with trim sides, only
    # retain the local max along the direction

    # Hence, keep the sharpest edge among a sea of edges
    def non_maximum_suppression(self, magnitude, direction):
        height, width = magnitude.shape

        output = np.zeors_like(magnitude)

        angle = np.degrees(direction)
        for row in range(height):
            for col in range(width):
                if angle[row,col] < 0:
                    angle[row,col] += 180

        for row in range(1, height - 1):
            for col in range(1, width - 1):

                # Intensity of two points along the direction
                q = 0
                r = 0

                angle_temp = angle[row, col]

                # 0 degrees and the vicinity
                if(
                    (0.0 <= angle_temp < 22.5)
                    or 
                    (157.5 < angle_temp <= 180.0)
                ):
                    q = angle[row, col - 1]
                    r = angle[row, col + 1]

                # 45 degrees and the vicinity
                if(
                    (22.5 <= angle_temp < 67.5)
                ):
                    q = angle[row + 1, col - 1]
                    r = angle[row - 1, col + 1]
                
                # 90 degrees and the vicinity
                if(
                    (67.5 <= angle_temp < 112.5)
                ):
                    q = angle[row + 1, col]
                    r = angle[row - 1, col]
                
                # 135 degrees and the vicinity
                if(
                    (112,5 <= angle_temp < 157.5)
                ):
                    q = angle[row - 1, col - 1]
                    r = angle[row + 1, col + 1]
        
        if (
            (magnitude[row, col] >= q)
            and
            (magnitude[row, col] >= r)
        ):
            output[row,col] = magnitude[row, col]

        return output

    def harris_response(
        self,
        image,
        gaussian_size=3,
        sigma=1.0,
        k=0.04
    ):
        # Step 1: gradients
        Ix = self.sobel_x(image)
        Iy = self.sobel_y(image)

        # Step 2: gradient products
        Ixx = Ix * Ix
        Iyy = Iy * Iy
        Ixy = Ix * Iy

        # Step 3: weighted local sums
        #
        # gaussian_blur already performs:
        # sum(w * local_patch)
        #
        # at every valid location.
        Sxx = self.gaussian_blur(
            Ixx,
            gaussian_size,
            sigma
        )

        Syy = self.gaussian_blur(
            Iyy,
            gaussian_size,
            sigma
        )

        Sxy = self.gaussian_blur(
            Ixy,
            gaussian_size,
            sigma
        )

        # Step 4: determinant and trace
        det = Sxx * Syy - Sxy**2

        trace = Sxx + Syy

        # Step 5: Harris response
        R = det - k * trace**2

        return R

