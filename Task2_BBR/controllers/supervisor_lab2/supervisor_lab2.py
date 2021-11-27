from controller import Supervisor
import sys
import random


class SupervisorLight:
    def __init__(self):
        # Simulation Parameters
        self.time_step = 32  # (ms)
        self.time_light = 20  # (s)
        self.flag_light = -1  # You can use the flag to identify the current position of the light node

        # Initiate Supervisor Module
        self.supervisor = Supervisor()
        # Get the robot node from your world environment
        self.robot_node = self.supervisor.getFromDef("Controller")
        # Check if the robot node exists 
        if self.robot_node is None:
            sys.stderr.write("No DEF Controller node found in the current world file\n")
            sys.exit(1)
        # Get the rotation and translation fields from your robot node
        self.trans_field = self.robot_node.getField("translation")
        self.rot_field = self.robot_node.getField("rotation")

        # Get the light node from your world environment
        self.light_node1 = self.supervisor.getFromDef("Light1")
        self.light_node2 = self.supervisor.getFromDef("Light2")
        self.light_node3 = self.supervisor.getFromDef("Light3")
        self.light_node4 = self.supervisor.getFromDef("Light4")
        self.light_list = [self.light_node1, self.light_node2, self.light_node3, self.light_node4]

    def run_seconds(self, seconds):
        # Calculate the number of iterations of the loop based on the time_step of the simulator 
        stop = int((seconds * 1000) / self.time_step)
        # Reset the counter
        iterations = 0
        # Run the loop and count the number of the iteration until it reaches the 'stop' value, which means 60 s 
        while self.supervisor.step(self.time_step) != -1:
            # This conditions is true after every 60 s 
            if stop == iterations:
                # Reset the counter
                iterations = 0
                # Reset physics of the robot (position and rotation)
                # Position
                INITIAL_TRANS = [0, 0, 0]
                self.trans_field.setSFVec3f(INITIAL_TRANS)
                # Rotation
                INITIAL_ROT = [0, 1, 0, -0.0]
                self.rot_field.setSFRotation(INITIAL_ROT)
                self.robot_node.resetPhysics()
                # Similar to the robot, you should change light position of the light node (self.light_node)
                # turn on a light by random
                random.shuffle(self.light_list)
                index = 0
                for i in self.light_list:
                    turn = i.getField("on")
                    if index == 0:
                        turn.setSFBool(True)
                    else:
                        turn.setSFBool(False)
                    index = index + 1

            # Increment the counter
            iterations = iterations + 1

    def run_demo(self):
        # Reset physics of the robot (position and rotation)
        # Position
        INITIAL_TRANS = [0, 0, 0]
        self.trans_field.setSFVec3f(INITIAL_TRANS)
        # Rotation
        INITIAL_ROT = [0, 1, 0, -0.0]
        self.rot_field.setSFRotation(INITIAL_ROT)
        self.robot_node.resetPhysics()
        # turn on a light by random
        random.shuffle(self.light_list)
        index = 0
        for i in self.light_list:
            turn = i.getField("on")
            if index == 0:
                turn.setSFBool(True)
            else:
                turn.setSFBool(False)
            index = index + 1
        # Update the position of the source of light after every 60 s (self.time_light == 60)
        self.run_seconds(self.time_light)


if __name__ == "__main__":
    # Create Supervisor Controller
    model = SupervisorLight()
    # Run Supervisor Controller
    model.run_demo()
