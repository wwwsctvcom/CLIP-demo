from PIL import Image
from transformers import CLIPModel, CLIPProcessor

if __name__ == "__main__":
    image_path = "./asset/dog.jpg"
    text = "this is a dog running"
    model_name_or_path = "./model_trained/best"
    processor = CLIPProcessor.from_pretrained(model_name_or_path)
    model = CLIPModel.from_pretrained(model_name_or_path)
    inputs = processor(text=text, images=Image.open(image_path), return_tensors="pt", padding=True)
    # output
    outputs = model(**inputs, return_loss=True)

    # loss
    print(outputs.loss)

    # similarity
    print(outputs.logits_per_image)  # [1, 3]
