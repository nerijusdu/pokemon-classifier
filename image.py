from PIL import Image, ImageOps
from pathlib import Path

def prepImage(path):
  img = Image.open(path).convert('RGB')
  img = ImageOps.fit(img, (300, 300), Image.ANTIALIAS, 0, (0.5, 0.5))
  img = ImageOps.grayscale(img)

  pathParts = path.split('/')
  newDir = 'preparedData/' + pathParts[1]
  newPath = newDir + '/' + pathParts[2]
  Path(newDir).mkdir(parents=True, exist_ok=True)

  img.save(newPath)
  return pathParts[2]