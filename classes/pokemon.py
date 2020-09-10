class Pokemon:
  def __init__(self, name):
    self.name = name
    self.images = []

  def print(self):
    print('Pokemon: ' + self.name)