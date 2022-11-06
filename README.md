# PokeGAN - Pokemon Sprite Creation via DCGAN
![Pokemon](images/charmander.png) ![Pokemon](images/charmeleon.png) ![Pokemon](images/charizard.png)   ![Pokemon](images/squirtle.png) ![Pokemon](images/wartortle.png) ![Pokemon](images/blastoise.png)  ![Pokemon](images/bulbasaur.png) ![Pokemon](images/ivysaur.png) ![Pokemon](images/venusaur.png) 

**This repo is used to scrape Pokemon sprite images from [https://pokemondb.net/sprites](https://pokemondb.net/sprites) and then use them to train a Generative Adversarial Network (GAN) to create new sprites.**

Sprites can be scraped using *scrape.py* and the DCGAN can then be trained using *train.py*. Generator weights are saved under **weights/** and images generated at each epoch are saved under **generated/**.