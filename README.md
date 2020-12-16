# my very cool project

i didn't label bc i suck

input   | camera motion  
depth   | scene flow

for all my friends viewing this
depth and input image are self explanatory (darker color means closer for depth)

the images on the right depict the motion in the scene

you can see that the camera motion fails at points where the ego-car is still but there's a lot of dynamic motion and the dense scene flow estimator fails in static scenes with a lot of texture-less regions

And that the other estimator succeeds where the other fails :-)

![Alt Text](./frames_ds.gif)
