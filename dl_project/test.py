from PIL import Image

def crop_center(pil_img, crop_width, crop_height):
    img_width, img_height = pil_img.size
    return pil_img.crop(((img_width - crop_width) // 2,
                         (img_height - crop_height) // 2,
                         (img_width + crop_width) // 2,
                         (img_height + crop_height) // 2))


image = Image.open("initial0.jpg")
print(image.size)
image = crop_center(image, 820,820)
print(image.size)
image.save("0.jpg")