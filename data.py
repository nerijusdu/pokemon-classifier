from os import listdir
from os.path import isfile, join
from classes.pokemon import Pokemon
import image

dataDir = 'dataset'
availableExtensions = ('jpg','jpeg','png')

def getPokemon():
  allPokemon = []

  for name in listdir(dataDir):
    pics = listdir(dataDir + '/' + name)
    for p in pics:
      if not p.endswith(availableExtensions):
        pics.remove(p)
    pokemon = Pokemon(name)
    pokemon.images = pics
    allPokemon.append(pokemon)
  
  return allPokemon

def prepImages(pokemonList):
  for pokemon in pokemonList:
    preparedImages = []
    print(pokemon.name)
    
    for img in pokemon.images:
      image.prepImage(dataDir + '/' + pokemon.name + '/' + img)
