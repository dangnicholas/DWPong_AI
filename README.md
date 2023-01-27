# Standalone AI for DWPONG

use the environment.yml alongside anaconda for the creation of an environment

## Unity to MQTT

- **Publish to MQTT**
   - camera/gamestate: 
      1. Publishes a single string of 0s (paddles and ball) and 1s (empty space) with 92160 digits (e.g. "0000011100...")
   - game/frame:
      1. Publishes a frame count as a string which is grabbed from Unity's Time.frameCount
   - game/level:
      1. Publishes a game level as string
- **Received from MQTT**
   - paddle1/action
      1. Receives a number from 0-2 (left, right, do nothing) based on inference
      
      
 ## AI to MQTT
 
- **Publish from MQTT**
   - paddle1/action
      1. Publishes a number from 0-2 (left, right, do nothing) based on inference
- **Received from MQTT**
   - camera/gamestate: 
      1. Receives a single string of 0s (paddles and ball) and 1s (empty space) with 92160 digits (e.g. "0000011100...")
      2. String is converted into a 160x192x3 np array for model input
   - game/frame:
      1. Receives a frame count as a string which is grabbed from Unity's Time.frameCount
   - game/level:
      1. Receives a game level as string
