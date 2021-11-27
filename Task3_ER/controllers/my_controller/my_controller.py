from controller import Robot
from datetime import datetime
import numpy as np
import csv
import os.path

class Controller:
    def __init__(self, robot):
        # Robot Parameters
        self.robot = robot
        self.time_step = 32  # times of basicTimeStep in WorldInfo
        self.rotation_speed = 6.28  # rad/s

        # Enable Motors
        self.left_motor = self.robot.getDevice('left wheel motor')
        self.right_motor = self.robot.getDevice('right wheel motor')
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)
        self.velocity_left = 0
        self.velocity_right = 0

        # Enable Proximity Sensors
        self.proximity_sensors = []
        for i in range(8):
            sensor_name = 'ps' + str(i)
            self.proximity_sensors.append(self.robot.getDevice(sensor_name))
            self.proximity_sensors[i].enable(self.time_step)

        # Enable Ground Sensors
        self.left_ir = self.robot.getDevice('gs0')
        self.left_ir.enable(self.time_step)
        self.center_ir = self.robot.getDevice('gs1')
        self.center_ir.enable(self.time_step)
        self.right_ir = self.robot.getDevice('gs2')
        self.right_ir.enable(self.time_step)

        # finite-state automaton
        # FSA[0]: detect object in the front
        # FSA[1]: turn in-place that making moving direction parallel to object surface
        # FSA[2]: move forward that parallel to object surface
        # FSA[3]: turn that making moving direction parallel to object surface
        # FSA[4]: follow the line
        # FSA[5]: rotate in-place to find the line again
        # FSA[6]: wander
        self.FSA = [1, 1, 2, 3, 4, 5, 6]
        self.current_state = 0

        self.fitness_values = []
        self.fitness = 0

        # Data
        self.inputs = []
        self.inputsPrevious = []
        self.distance_max = 0.07
        # Flag
        self.side = ""

    def clip_value(self, value, min_max):
        if value > min_max:
            return min_max
        elif value < -min_max:
            return -min_max
        return value

    def run_robot(self, filename):
        # Main Loop
        count = 0
        inputs_avg = []
        data = []
        while self.robot.step(self.time_step) != -1:
            # Read Ground Sensors
            self.inputs = []
            left = self.left_ir.getValue()
            center = self.center_ir.getValue()
            right = self.right_ir.getValue()
            # print(str(left)+" "+str(center)+" "+ str(right))
            # if black 300, if white >750
            # Adjust Values
            min_gs = 0
            max_gs = 1000
            if left > max_gs:
                left = max_gs
            if center > max_gs:
                center = max_gs
            if right > max_gs:
                right = max_gs
            if left < min_gs:
                left = min_gs
            if center < min_gs:
                center = min_gs
            if right < min_gs:
                right = min_gs

            # Save Data
            self.inputs.append((left - min_gs) / (max_gs - min_gs))
            self.inputs.append((center - min_gs) / (max_gs - min_gs))
            self.inputs.append((right - min_gs) / (max_gs - min_gs))
            print("Ground Sensors \n    left {} center {} right {}".format(self.inputs[0], self.inputs[1], self.inputs[2]))

            # Read Distance Sensors
            for i in range(8):
                if i == 0 or i == 1 or i == 2 or i == 5 or i == 6 or i == 7:
                    temp = self.proximity_sensors[i].getValue()
                    # Adjust Values
                    min_ds = 0
                    max_ds = 1000
                    if temp > max_ds:
                        temp = max_ds
                    if temp < min_ds:
                        temp = min_ds
                    # Save Data
                    self.inputs.append((temp - min_ds) / (max_ds - min_ds))
                    print("Distance Sensors - Index: {}  Value: {}".format(i,self.proximity_sensors[i].getValue()))

            # Smooth filter (Average)
            smooth = 1  # average of two
            if count == smooth:
                inputs_avg = [sum(x) for x in zip(*inputs_avg)]
                self.inputs = [x / smooth for x in inputs_avg]
                self.inputsPrevious = self.inputs
                # Compute and actuate
                tempv = self.sense_compute_and_actuate()
                self.inputs.append(tempv[0])
                self.inputs.append(tempv[1])
                tempf = self.calculate_fitness()
                self.inputs = np.hstack((self.inputs, tempf))
                file_exists = os.path.isfile(str(filename))
                with open("results_task3_sensor_BBR.csv", "a", newline='') as file:
                    header = ["Ground sensor left", "Ground sensor middle", "Ground sensor right", "proximity sensor 0",
                              "proximity sensor 1", "proximity sensor 2", "proximity sensor 5", "proximity sensor 6",
                              "proximity sensor 7", "velocity left", "velocity right", "forward fitness",
                              "follow line fitness", " avoid collision fitness", " spinning fitness", "combined fitness"]
                    writer = csv.writer(file)

                    if not file_exists:
                        # prints header into file if file doesn't exist yet, otherwise just prints results as row.
                        writer.writerow(header)

                    writer.writerow(self.inputs)
                    writer.writerow(self.inputs)
                # Reset
                count = 0
                inputs_avg = []

            else:
                inputs_avg.append(self.inputs)
                count = count + 1

    def sense_compute_and_actuate(self):
        if len(self.inputs) > 0 and len(self.inputsPrevious) > 0:
            # threshold of detecting a obstacle
            if max(self.inputs[3], self.inputs[4], self.inputs[7], self.inputs[8]) >= 0.15:
                self.current_state = self.FSA[0]
            # while rotate in place, the value of proximity sensor 2 or 5 will increase first
            elif self.current_state == 1 and (self.inputs[5] > self.distance_max or self.inputs[6] > self.distance_max):
                # if the obstacle is on the right-hand side
                if self.side == "right":
                    if self.inputs[5] > self.distance_max:
                        self.current_state = self.FSA[1]
                        # updating the maximum value of the distance between proximity sensor 2 or 5 with the obstacle
                        self.distance_max = self.inputs[5]
                # if the obstacle is on the left-hand side
                elif self.side == "left":
                    if self.inputs[6] > self.distance_max:
                        self.current_state = self.FSA[1]
                        self.distance_max = self.inputs[6]
            # threshold of detecting the line
            elif np.min(self.inputs[0:3]) < 0.38:
                self.current_state = self.FSA[4]
                # reset the maximum of distance
                self.distance_max = 0.07
            # can't find the line, rotate in-place
            elif (self.current_state == 4 or self.current_state == 5) and np.max(self.inputs[0:3]) > 0.6:
                self.current_state = self.FSA[5]
            # at the edge of the obstacle, has to turn
            elif (self.current_state == 2 or self.current_state == 3) \
                    and ((self.side == "right" and self.inputs[5] < 0.08)
                         or (self.side == "left" and self.inputs[6] < 0.08)):
                self.current_state = self.FSA[3]
            # when the distance between proximity sensor 2 or 5 with the obstacle start decreasing, stop turn, and move forward
            elif (self.current_state == 1 or self.current_state == 3) \
                    and (self.inputs[5] < self.distance_max
                         or self.inputs[6] < self.distance_max):
                if self.side == "right":
                    if self.inputs[5] < self.distance_max:
                        self.current_state = self.FSA[2]
                elif self.side == "left":
                    if self.inputs[6] < self.distance_max:
                        self.current_state = self.FSA[2]
            # nothing in the front, wandering
            elif np.max(self.inputs[3:9]) < 0.08:
                self.current_state = self.FSA[6]
        print(self.current_state)
        if self.current_state == 0:
            pass

        elif self.current_state == 1:
            max_direction = max(self.inputs[3], self.inputs[4], self.inputs[7], self.inputs[8])
            # if the obstacle still in the front
            if max_direction >= 0.15:
                # if the obstacle is on the right-hand side
                if max_direction == max(self.inputs[3], self.inputs[4]):
                    self.side = "right"
                # if the obstacle is on the left-hand side
                else:
                    self.side = "left"
            # if on the right-hand side, rotate anticlockwise. vice versa
            if self.side == "right":
                self.velocity_left = -self.rotation_speed / 2
                self.velocity_right = self.rotation_speed / 2
            elif self.side == "left":
                self.velocity_left = self.rotation_speed / 2
                self.velocity_right = -self.rotation_speed / 2

        elif self.current_state == 2:
            if self.side == "right":
                # if the side proximity sensor decreased, move right slightly, vice versa
                if self.inputs[5] < self.distance_max:
                    self.velocity_left = self.rotation_speed / 1.8
                    self.velocity_right = self.rotation_speed / 2
                elif self.inputs[5] >= self.distance_max:
                    self.velocity_left = self.rotation_speed / 2
                    self.velocity_right = self.rotation_speed / 1.8

            elif self.side == "left":
                # vice versa
                if self.inputs[6] < self.distance_max:
                    self.velocity_left = self.rotation_speed / 2
                    self.velocity_right = self.rotation_speed / 1.8
                elif self.inputs[6] >= self.distance_max:
                    self.velocity_left = self.rotation_speed / 1.8
                    self.velocity_right = self.rotation_speed / 2

        elif self.current_state == 3:
            # turn right immediately
            if self.side == "right":
                self.velocity_left = self.rotation_speed / 4
                self.velocity_right = 0.1
            # vice versa
            elif self.side == "left":
                self.velocity_left = 0.1
                self.velocity_right = self.rotation_speed / 4

        elif self.current_state == 4:
            # the left ground sensor is the smallest (most black one), go left slightly
            if self.inputs[0] < self.inputs[1] and self.inputs[0] < self.inputs[2]:
                self.velocity_left = 1.4
                self.velocity_right = 2
            # the middle ground sensor is the smallest (most black one), go straight forward
            elif self.inputs[1] < self.inputs[0] and self.inputs[1] < self.inputs[2]:
                self.velocity_left = 2
                self.velocity_right = 2
            # the right ground sensor is the smallest (most black one), go right slightly
            elif self.inputs[2] < self.inputs[0] and self.inputs[2] < self.inputs[1]:
                self.velocity_left = 2
                self.velocity_right = 1.4

        elif self.current_state == 5:
            # if previous obstacle is on the right-hand side, we have to rotate anticlockwise, otherwise, we will meet
            # that obstacle again
            if self.side == "right":
                self.velocity_left = -4
                self.velocity_right = 4
            # vice versa
            elif self.side == "left":
                self.velocity_left = 4
                self.velocity_right = -4

        elif self.current_state == 6:
            self.velocity_left = np.random.rand() * 2
            self.velocity_right = np.random.rand() * 2

        self.left_motor.setVelocity(self.velocity_left)
        self.right_motor.setVelocity(self.velocity_right)

        return [self.velocity_left, self.velocity_right]

    def calculate_fitness(self):
        ### Define the fitness function to increase the speed of the robot and
        ### to encourage the robot to move forward only
        # base value is the 7 times of the sum of two speed
        forwardFitness = (self.velocity_left + self.velocity_right) *7
        # the penalty is the difference between two speed, if the difference smaller than 2, then no penalty
        if abs(self.velocity_right - self.velocity_left) < 2:
            forwardFitness = forwardFitness
        # if greater than 2, the base value will be divided by the difference value
        else:
            forwardFitness = forwardFitness / abs(self.velocity_right - self.velocity_left)


        ### Define the fitness function to encourage the robot to follow the line
        # at least one of the ground sensors has to detect the line
        followLineFitness = 100 * (0.7-np.min(self.inputs[0:3]))

        ### Define the fitness function to avoid collision
        # due to that not every time there is a obstacle, so, the fitness of this as multiple of the final fitness
        # can't too close with a obstacle
        if np.max(self.inputs[3:9]) >= 0.2075:
            avoidCollisionFitness = 0.001
        # when detect a obstacle on the right-hand side, turn left significantly
        elif (0.07799 < self.inputs[3] < 0.2075 or 0.07799 < self.inputs[4] < 0.2075) and \
                (self.velocity_right > self.velocity_left) and self.velocity_right - self.velocity_left >= 1.5:
            avoidCollisionFitness = 1.3
        # when detect a obstacle on the left-hand side, turn right significantly
        elif (0.07799 < self.inputs[7] < 0.2075 or 0.07799 < self.inputs[8] < 0.2075) and \
                (self.velocity_right < self.velocity_left) and self.velocity_left - self.velocity_right >= 1.5:
            avoidCollisionFitness = 1.3
        # when parallel with the obstacle, moving forward with slight curve
        elif (0.07799 < self.inputs[5] < 0.2075 or 0.07799 < self.inputs[6] < 0.2075) and \
                abs(self.velocity_right - self.velocity_left) < 1.5:
            avoidCollisionFitness = 1.3
        # no obstacle case
        elif np.max(self.inputs[3:9]) < 0.07799:
            avoidCollisionFitness = 1
        else:
            avoidCollisionFitness = 0.5

        ### Define the fitness function to avoid spining behaviour
        # simply can't have a negative speed
        if self.velocity_left < 0 or self.velocity_right < 0:
            spinningFitness = 0
        else:
            spinningFitness = 100
        # print("forward fitness {0} ","follow line fit {1}", " avoid collision fit {2}", " spinning fit {3} ",{forwardFitness},{followLineFitness},{avoidCollisionFitness},{spinningFitness})
        ### Define the fitness function of this iteration which should be a combination of the previous functions
        # following the line has highest priority, and then is the speed, then the spinning
        combinedFitness = (forwardFitness * 0.5 + followLineFitness * 0.6 + spinningFitness * 0.4) * avoidCollisionFitness

        self.fitness_values.append(combinedFitness)
        self.fitness = np.mean(self.fitness_values)

        return [forwardFitness, followLineFitness, avoidCollisionFitness, spinningFitness, combinedFitness]

if __name__ == "__main__":
    my_robot = Robot()
    controller = Controller(my_robot)
    controller.run_robot("results_task3_sensor_BBR.csv")

