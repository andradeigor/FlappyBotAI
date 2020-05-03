# FlappyBotAI
This is the first code that I wrote, so, probably has a lot of mistakes and no usual (dumb) way to do things. 
I know that, but anyway I have to put this online to save my progress. In the future, I hope to look at this and see how bad 
I was and how much I got better. 


I wrote this inspired by some videos that I watched online(Tech With Tim, Universo Programado, etc...), I think this would be a good way 
to start coding and learning some AI stuff. I tried not to just copy the way these guys did their own codes, and made this more difficult 
for me. I made a Flappy Bird, just like them, but I implement the AI in another code. Because of that, I have to find a way to get the 
object's coordinates from the game. The way that I got this was using Open-Cv, using Template Matching and Color Search I can find the 
bird and the pipes coordinates. After that, I send this coordinates to my Neural Network with genetic evolution(NEAT) and it learn how to play.
