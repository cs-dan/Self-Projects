# mnistGAN Notes

*Running the program
It's a script, just run it and let it run its course.

*What does it do? 
Its a smol GAN model to produce 28x28 images of seemingly handwritten numbers.

*Why?
I wanted to learn.

*Special Mention-
Many thanks to the guys over at machinelearningmastery.com. The GAN structure and a lot of the code was adapted from their tutorial.

*Notes on the process:
**General
- GAN consists of a discriminator and a generator. One is to train on what real images are like. The later then compares against the discrim when generating values for the produced image. 

**Regarding the Generator
- So the generator is like an inverse CNN? That's pretty cool. 
- The generator is not trained directly. So no loss func or optimizer is specified. 
- LeakyReLU with a slope of 0.2 is considered good practice for GANs for some reason.
- The relationship between the discriminator and the generator is adversarial, ie the generator is updated more when the discriminator is good at discerning the fakes from the real and vice versa when it isn't.
- In putting it all together, we'll start off with random inputs to the generator, of which makes samples (generated images) that are fed to the discriminator who then updates the weights of the generator based on the performance of the discriminator. 

