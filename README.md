# Mars Lander - Open ai enviroment

Made By Michi, Zied and Zhen

## About
mars landing an Open ai test environment.
About this environment:
The goal is to land at the base station with the satelite, on the way to the station the satelite must avoid obstacles like black holes or asteroids.

The satelite starts from an random position somewhere in the upper lefter corner and flies with an constant speed in direction of the station.

It contains the standart Open AI enviroment methods :
- reset()
- step()
 We hope you guys know how to use these

 The enviroment is solved if you hit the space station five times in a row. A flag will be set if this is the case (there will also be a console output). The flag is part of step return (info == True) if solved.

### Important for usage!
 For setting the level of dificulty init the nviroment like this :
 ```python
  env = gym.make('mars_lander-v0', level=1)
 ```
 where level is 1, 2 or 3
 

 ALSO you have to render the env youself add something like this to your agent if you want
 ```python
  if render:
      still_open = self.env.render()
      if still_open is False:
          break
 ```

## Observations

This environment consists of several levels. At level 1 there are no obstacles, at level 2 there are obstacles and at level 3 there are obstacles that always appear and move around.
for level 1 is the observation space:
- s[0] x position of the satelite
- s[1] y position of the saddle rivet
- s[2] x position of the base station
- s[3] y position of the base station

for level 2 and three the positions of the obstacles are transferred to the observations of 1
- s[4] x position of the asteroid
- s[5] y position of the asteroid
- s[6] x position of the black hole_1
- s[7] y position of the black hole_1
- s[8] x position of the black hole_2
- s[9] y position of the black hole_2

## Actions

The possible actions of the satelite are discrete, either it activates the motors (action = 1) or it deactivates them (action = 0).

## Reward
The reward is sparse:
(1) For hitting the space station the agent gets +50
(2) For hitting an obstacle he gets -50
(3) For crushing in the ground he gets 25 - distance to the lander

## Virtual enviroment
How to use : 
- First you have to init a venv on your desktop we do not want to push it on git for this run:
  ```shell
  python3 -m venv venv
  ```
- After that activate venv by running and install pip-tools
  ```shell
  source venv/bin/activate
  python3 -m pip install pip-tools
  ```
- Then compile requirements.in with command line :
  ```shell
  pip-compile dependencies/requirements.in
  ```
- pip-compile requirements.in  --> CREATES requirements.txt
- this file can be used in venv to install all requirements
  ```shell
  pip install -r dependencies/requirements.txt
  ```
