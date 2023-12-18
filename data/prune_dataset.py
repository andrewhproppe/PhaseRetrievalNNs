import cv2
import os

def display_image(image, scale_factor=2.0):
    height, width, _ = image.shape
    new_height = int(height * scale_factor)
    new_width = int(width * scale_factor)
    enlarged_image = cv2.resize(image, (new_width, new_height))
    cv2.imshow('Image', enlarged_image)

def crop_image(image, crop_percentage=0.1):
    height, width, _ = image.shape
    crop_margin_h = int(height * crop_percentage / 2)
    crop_margin_w = int(width * crop_percentage / 2)
    cropped_image = image[crop_margin_h:height-crop_margin_h, crop_margin_w:width-crop_margin_w]
    return cropped_image

def save_image(image, path):
    cv2.imwrite(path, image)

def main(directory):
    image_files = sorted([f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))])

    # Remove any files that have already been processed, which are denoted by a '_p' in the filename
    image_files = [f for f in image_files if '_p' not in f]

    for image_file in image_files:
        image_path = os.path.join(directory, image_file)
        original_image = cv2.imread(image_path)

        if original_image is not None:
            display_image(original_image)
            key = cv2.waitKey(0)

            if key == ord('k'):
                print(f"Keeping {image_file}")
                new_image_path = os.path.join(directory, image_file.replace('.', '_p.'))
                save_image(original_image, new_image_path)
                os.remove(image_path)
            elif key == ord('d'):
                os.remove(image_path)
                print(f"Deleted {image_file}")
            elif key == ord('e'):
                print('Exiting program')
                break
            elif key == ord('z'):
                cropped_image = original_image.copy()
                while True:
                    display_image(cropped_image)
                    crop_key = cv2.waitKey(0)

                    if crop_key == ord('z'):
                        cropped_image = crop_image(cropped_image, crop_percentage=0.1)
                    elif crop_key == ord('k'):
                        print(f"Keeping cropped {image_file}")
                        new_image_path = os.path.join(directory, image_file.replace('.', '_p.'))
                        save_image(cropped_image, new_image_path)
                        os.remove(image_path)
                        break
                    elif crop_key == ord('d'):
                        print(f"Deleted cropped {image_file}")
                        break

            cv2.destroyAllWindows()


if __name__ == "__main__":
    directory_path = "masks/flowers_pruned"
    main(directory_path)