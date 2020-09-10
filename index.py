import data
import sys

shouldPrepare = len(sys.argv) > 1 and (sys.argv[1] == '--prepare' or sys.argv[1] == '-p')

allPokemon = data.getPokemon()
if shouldPrepare:
  data.prepImages(allPokemon)


print('end')