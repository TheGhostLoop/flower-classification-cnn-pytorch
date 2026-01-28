from PIL import Image 

img = Image.open("test_image_bellflower.jpg").convert("RGB")
img = transform(img)
img = img.unsqueeze(0).to(device)
dataset_classes = train_dataset.classes 

with torch.no_grad():
  model.eval()
  logits = model(img)
  pred_idx = torch.argmax(logits,dim=1)
  print(f"Flower is:: {dataset_classes[pred_idx]}")
