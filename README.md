# PokeGAN - Pokemon Sprite Creation via DCGAN
![Pokemon](sprites/charmander.png) ![Pokemon](sprites/charmeleon.png) ![Pokemon](sprites/charizard.png)   ![Pokemon](sprites/squirtle.png) ![Pokemon](sprites/wartortle.png) ![Pokemon](sprites/blastoise.png)  ![Pokemon](sprites/bulbasaur.png) ![Pokemon](sprites/ivysaur.png) ![Pokemon](sprites/venusaur.png) 

**This repo is used to scrape Pokemon sprite images from [https://pokemondb.net/Pokemon](https://pokemondb.net/Pokemon) and then use them to train a Generative Adversarial Network (GAN) to create new Pokemon.**

Pokemon can be scraped using *scrape.py* and the DCGAN can then be trained using *train.py*. Generator weights are saved under **weights/** and images generated at each epoch are saved under **generated/**.