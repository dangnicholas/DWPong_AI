import sys
import os
from logging.handlers import RotatingFileHandler

from exhibit.ai.model import PGAgent
from exhibit.shared.config import Config
#from exhibit.game.game_subscriber import GameSubscriber
import time
from exhibit.ai.ai_subscriber import AISubscriber
import numpy as np
import cv2
import threading
from exhibit.shared.utils import Timer

from queue import Queue

# NICK - for windows
import os
from PIL import Image

import logging

class AIDriver:
    # #MODEL = 'validation/canstop_randomstart_6850.h5'#'../../validation/newhit_10k.h5'
    # MODEL_1 = f'./validation/canstop_randomstart_3k.h5'
    # MODEL_2 = f'./validation/canstop_randomstart_6850.h5'
    # MODEL_3 = f'./validation/canstop_randomstart_10k.h5'

    # The locations of the three models used. 1 for each level. Works for Windows and Linux
    root_dirname = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    MODEL_1 = os.path.join(root_dirname, 'validation', 'smoothreward_s6_f5_d3_5000.h5')
    MODEL_2 = os.path.join(root_dirname, 'validation', 'smoothreward_s6_f5_d3_15000.h5')
    MODEL_3 = os.path.join(root_dirname, 'validation', 'smoothreward_s6_f5_d3_22850.h5')
    level = 0

    def publish_inference(self):
        """
        This method will publish some its inference from the game state it has
        and publish action and frame to paddle1 or paddle2 based on which the AI
        is to control. Will also check if the level changed and load a new model if needed
        """

        # Check if the level has changed. If so, we need to load a new model
        if (AIDriver.level != self.state.game_level):
            # check if a kill/quit message has been sent via the queue
            if not self.q.empty():
                dataQ = self.q.get()
                if dataQ == "endThreads":
                    self.logger.info('ai thread quitting')
                    while not self.q.empty:  # empty the rest of the q
                        dataQ = self.q.get()
                    self.q.put('noneActive')
                    # a message that goes back to the main program to tell it that the ai_driver has stopped
                    sys.exit()
                    self.logger.error('the sys exit didnt work')

            temp = AIDriver.level
            AIDriver.level = self.state.game_level
            self.logger.info(f'level changed to {AIDriver.level}')
            if self.state.game_level == 0:
                self.agent = self.agent1
                # self.agent1.load(AIDriver.MODEL_1)
            elif self.state.game_level == 1 and temp != 0:
                self.agent = self.agent1
                # self.agent1.load(AIDriver.MODEL_1)
            elif self.state.game_level == 2:
                self.agent = self.agent2
                # self.agent1.load(AIDriver.MODEL_2)
            elif self.state.game_level == 3:
                self.agent = self.agent3
                # self.agent1.load(AIDriver.MODEL_3)
            else:
                self.agent = self.agent1
                # self.agent1.load(AIDriver.MODEL_2)

        diff_state = self.state.render_latest_diff()
        # Get latest state diff

        current_frame_id = self.state.frame
        if isinstance(diff_state, np.ndarray):
            # Infer on flattened state vector
            x = diff_state.ravel()
            action, _, probs = self.agent.act(x, greedy=True)

            # Stores the first 500 frames fed into AI model for debugging if needed
            if self.last_acted_frame < 500:
                img = Image.fromarray(diff_state.astype('uint8'), 'L')
                img.save(os.path.join(self.root_dirname, 'logs', 'ai_input_images', f"{self.last_acted_frame}_{action}.png"))
            # Publish prediction
            if self.paddle1:
                self.state.publish("paddle1/action", str(action))
                self.state.publish("paddle1/frame", str(current_frame_id))
                # self.logger.info("paddle1 action: " + str(action) + " | frame: " + str(current_frame_id))
            elif self.paddle2:
                self.state.publish("paddle2/action", str(action))
                self.state.publish("paddle2/frame", str(current_frame_id))
                # self.logger.info("paddle2 action:" + str(action) + " | frame: " + str(current_frame_id))

            model_activation = self.agent.get_activation_packet()
            self.state.publish("ai/activation", model_activation)




    def inference_loop(self):
        """
        This method will loop indefinitely to check whether to run inference
        based on whether the game state it has is updated
        """
        while True:
            current_frame_id = self.state.frame
            if self.last_acted_frame == current_frame_id:
                time.sleep(0.001)
            else:
                self.publish_inference()
                self.last_acted_frame = current_frame_id

    def __init__(self, config=Config.instance(), paddle1=True, in_q=Queue()):
        """
        This method will construct the ai driver to load in all 3 models from the directories at the
        top of this page and create threads and an ai_subscriber object to listen over MQTT for game
        state passed over it
        """
        # Set up the logger
        # Creating log folders if not exists
        if not os.path.exists(os.path.join(self.root_dirname, 'logs')):
            os.makedirs(os.path.join(self.root_dirname, 'logs'))
            os.makedirs(os.path.join(self.root_dirname, 'logs', 'ai_input_images'))
            with open(os.path.join(self.root_dirname, 'logs', 'ai_driver.log'), 'w'): pass

        self.root_dirname = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler = RotatingFileHandler(os.path.join(self.root_dirname, "logs", "ai_driver.log"), maxBytes=5000, backupCount=10)
        self.logger = logging.getLogger(__name__)
        self.logger.addHandler(handler)
        self.logger.info("Logging set up for ai_driver")

        self.q = in_q
        self.config = config
        self.paddle1 = paddle1
        self.paddle2 = not self.paddle1

        # We have all 3 agents already loaded instead of loading between levels. Saves a lot of time and prevents freezing
        self.agent1 = PGAgent(self.config.CUSTOM_STATE_SIZE, self.config.CUSTOM_ACTION_SIZE)
        self.agent1.load(AIDriver.MODEL_1)
        self.agent = self.agent1
        self.agent2 = PGAgent(self.config.CUSTOM_STATE_SIZE, self.config.CUSTOM_ACTION_SIZE)
        self.agent2.load(AIDriver.MODEL_2)
        self.agent3 = PGAgent(self.config.CUSTOM_STATE_SIZE, self.config.CUSTOM_ACTION_SIZE)
        self.agent3.load(AIDriver.MODEL_3)
        self.state = AISubscriber(self.config, trigger_event=lambda: self.publish_inference())
        self.last_frame_id = self.state.frame
        self.last_tick = time.time()
        self.frame_diffs = []
        self.last_acted_frame = 0
        self.inference_thread = threading.Thread(target=self.inference_loop)
        self.inference_thread.start()
        self.state.start()


def main(in_q):
    """
    This will get the configs used in the config file and also construct the ai_driver using the constructor above
    """
    # main is separated out so that we can call it and pass in the queue from GUI
    config = Config.instance()
    instance = AIDriver(config=config, in_q=in_q)


if __name__ == "__main__":
    main(Queue())

