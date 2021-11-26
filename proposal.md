# Final Project Proposal - Shadow Mapping

Julio Grillo

## Overview

Shadow mapping is a technique for producing shadows in a scene without
full-blown raytracing (meaning it can be done within OpenGL). The technique
takes two steps. The first step is to render the scene from the
light's perspective, saving the depth values to a texture (know as the
depth map). Then, in the final rendering of the scene from camera
perspective, using the depth map to decide if a fragment is in-shadow or not.

## Timeline

[x] - 11/23 Basic shadow mapping technique
[ ] - 12/1 Demo
[ ] - 12/8 Minor improvements (ex. Sampling)
