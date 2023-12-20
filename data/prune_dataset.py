import cv2
import os
import numpy as np

def display_image(image, zoom=1, dpi=150):
    height, width, _ = image.shape

    zoomed_height = int(height*zoom)
    zoomed_width = int(width*zoom)

    height_diff = (height - zoomed_height)
    width_diff = (width - zoomed_width)

    zoomed_image = image[height_diff//2:-(height_diff//2)-1, width_diff//2:-(width_diff//2)-1, :]

    cv2.imshow('Image', zoomed_image)

    return zoomed_height, zoomed_width

def crop_center(image, zoom):
    height, width, _ = image.shape

    # Calculate crop margins
    crop_margin_h = int((1 - zoom) * height / 2)
    crop_margin_w = int((1 - zoom) * width / 2)

    # Crop the image
    cropped_image = image[crop_margin_h:height-crop_margin_h, crop_margin_w:width-crop_margin_w]

    return cropped_image

def save_image(image, path):
    cv2.imwrite(path, image)

def main(directory):
    image_files = sorted([f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))])

    # Remove any files that have already been processed, which are denoted by a '_p' in the filename
    image_files = [f for f in image_files if '_p' not in f]

    total_saved_images = 0
    total_deleted_images = 0

    for image_file in image_files:
        image_path = os.path.join(directory, image_file)
        original_image = cv2.imread(image_path)
        image = original_image.copy()

        if original_image is not None:
            display_image(original_image)
            original_height, original_width, _ = original_image.shape
            key = cv2.waitKey(0)

            if key == ord('k'):
                print(f"Keeping {image_file}")
                new_image_path = os.path.join(directory, image_file.replace('.', '_p.'))
                save_image(original_image, new_image_path)
                os.remove(image_path)
                total_saved_images += 1

            elif key == ord('d'):
                os.remove(image_path)
                print(f"Deleted {image_file}")
                total_deleted_images += 1

            elif key == ord('e'):
                print('Exiting program')
                break

            elif key == ord('z'):
                """ Start the 'cropping' loop """
                zoom = 0.9
                new_height, new_width = display_image(image, zoom=zoom)
                print(f"Cropped image dimensions: {(new_height, new_width)}")
                while True:
                    crop_key = cv2.waitKey(0)

                    if crop_key == ord('z'):
                        zoom -= 0.1

                    elif crop_key == ord('k'):
                        print(f"Keeping cropped {image_file}")
                        new_image_path = os.path.join(directory, image_file.replace('.', '_p.'))

                        new_image = crop_center(image, zoom)

                        save_image(new_image, new_image_path)
                        os.remove(image_path)
                        total_saved_images += 1
                        break

                    elif crop_key == ord('d'):
                        print(f"Deleted cropped {image_file}")
                        break

                    elif crop_key == ord('u'):
                        print(zoom)
                        if zoom >= 1:
                            print('Zoom at 1')
                        else:
                            zoom += 0.1

                    elif crop_key == ord('4'):
                        image = np.roll(image, -10, 1)
                    elif crop_key == ord('6'):
                        image = np.roll(image, +10, 1)
                    elif crop_key == ord('8'):
                        image = np.roll(image, +10, 0)
                    elif crop_key == ord('5'):
                        image = np.roll(image, -10, 0)

                    elif crop_key == ord('e'):
                        print('Exiting program')
                        return

                    new_height, new_width = display_image(image, zoom=zoom)
                    print(f"Image dimensions: {(new_height, new_width)}")



            print(f"Total images saved: {total_saved_images} | Total images deleted: {total_deleted_images}")

            cv2.destroyAllWindows()


if __name__ == "__main__":
    directory_path = "masks/flowers_more_pruned"
    main(directory_path)