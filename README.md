# PokeGAN - Pokemon Sprite Creation via DCGAN
![Sprites](images/charmander.png) ![Sprites](images/charmeleon.png) ![Sprites](images/charizard.png)   ![Sprites](images/squirtle.png) ![Sprites](images/wartortle.png) ![Sprites](images/blastoise.png)  ![Sprites](images/bulbasaur.png) ![Pokemon](images/ivysaur.png) ![Pokemon](images/venusaur.png) 

**This repo is used to scrape Pokemon sprite images from [https://pokemondb.net/sprites](https://pokemondb.net/sprites) and then use them to train a Generative Adversarial Network (GAN) to create new sprites.**

Sprites can be scraped using *scrape.py* and the DCGAN can then be trained using *train.py*. Generator weights are saved under **weights/** and images generated at each epoch are saved under **generated/**.