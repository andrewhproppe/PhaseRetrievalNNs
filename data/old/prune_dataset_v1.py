import cv2
import os

def display_image(image, scale_factor=1.0):
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

    total_saved_images = 0
    total_deleted_images = 0

    for image_file in image_files:
        image_path = os.path.join(directory, image_file)
        original_image = cv2.imread(image_path)
        crop_states = [original_image.copy()]

        if original_image is not None:
            display_image(original_image)
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
                cropped_image = original_image.copy()
                while True:
                    display_image(cropped_image)
                    crop_key = cv2.waitKey(0)

                    if crop_key == ord('z'):
                        cropped_image = crop_image(cropped_image, crop_percentage=0.1)
                        crop_states.append(cropped_image.copy())
                        print(f"Cropped image dimensions: {cropped_image.shape}")
                    elif crop_key == ord('k'):
                        print(f"Keeping cropped {image_file}")
                        new_image_path = os.path.join(directory, image_file.replace('.', '_p.'))
                        save_image(cropped_image, new_image_path)
                        os.remove(image_path)
                        total_saved_images += 1
                        break
                    elif crop_key == ord('d'):
                        print(f"Deleted cropped {image_file}")
                        break
                    elif crop_key == ord('u') and len(crop_states) > 1:
                        crop_states.pop()  # Remove the last state
                        cropped_image = crop_states[-1].copy()
                        print(f"Undone cropping. Current dimensions: {cropped_image.shape}")
                    elif key == ord('e'):
                        print('Exiting program')
                        break
            print(f"Total images saved: {total_saved_images} | Total images deleted: {total_deleted_images}")

            cv2.destroyAllWindows()

if __name__ == "__main__":
    directory_path = "masks/flowers_more_pruned"
    main(directory_path)