from controller import Robot
from datetime import datetime
import math
import numpy as np
import csv
import os.path


class Controller:
    def __init__(self, robot):
        # Robot Parameters
        self.robot = robot
        self.time_step = 32  # ms
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

        # Enable Distance Sensors
        self.distance_sensors = []
        for i in range(8):
            sensor_name = 'ps' + str(i)
            self.distance_sensors.append(self.robot.getDevice(sensor_name))
            self.distance_sensors[i].enable(self.time_step)

        # Enable Light Sensors
        self.light_sensors = []
        for i in range(8):
            sensor_name = 'ls' + str(i)
            self.light_sensors.append(self.robot.getDevice(sensor_name))
            self.light_sensors[i].enable(self.time_step)

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
        # FSA[4]: toward the light source
        # FSA[5]: reach the boundary
        # FSA[6]: wandering
        self.FSA = [1, 1, 2, 3, 4, 5, 6]
        self.current_state = 0

        # Data
        self.inputs = []
        self.inputsPrevious = []
        self.distance_max = 0.07

        # Flag
        self.side = "right"

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
        while self.robot.step(self.time_step) != -1:
            # Read Ground Sensors
            self.inputs = []
            left = self.left_ir.getValue()
            center = self.center_ir.getValue()
            right = self.right_ir.getValue()

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
            # print("Ground Sensors \n    left {} center {} right {}".format(self.inputs[0], self.inputs[1],
                                                                           # self.inputs[2]))

            # Read Light Sensors
            for i in range(8):
                temp = self.light_sensors[i].getValue()
                # Adjust Values
                min_ls = 0
                max_ls = 4200
                if temp > max_ls:
                    temp = max_ls
                if temp < min_ls:
                    temp = min_ls
                # Save Data
                self.inputs.append((temp - min_ls) / (max_ls - min_ls))
                # print("Light Sensors - Index: {}  Value: {}".format(i,self.light_sensors[i].getValue()))

            # Read Distance Sensors
            for i in range(8):
                temp = self.distance_sensors[i].getValue()
                # Adjust Values
                min_ls = 0
                max_ls = 1000
                if temp > max_ls:
                    temp = max_ls
                if temp < min_ls:
                    temp = min_ls
                # Save Data
                self.inputs.append((temp - min_ls) / (max_ls - min_ls))
                # print("Distance Sensors - Index: {}  Value: {}".format(i,self.distance_sensors[i].getValue()))

            smooth = 1  # average of two
            if count == smooth:
                inputs_avg = [sum(x) for x in zip(*inputs_avg)]
                self.inputs = [x / smooth for x in inputs_avg]
                self.inputsPrevious = self.inputs
                # Compute and actuate
                tempv = self.sense_compute_and_actuate()
                self.inputs.append(tempv[0])
                self.inputs.append(tempv[1])
                file_exists = os.path.isfile(str(filename))
                with open("results_task2.csv", "a", newline='') as file:
                    header = ["Ground sensor left", "Ground sensor middle", "Ground sensor right", "Light sensor 0",
                              "Light sensor 1", "Light sensor 2", "Light sensor 3", "Light sensor 4", "Light sensor 5",
                              "Light sensor 6", "Light sensor 7", "proximity sensor 0", "proximity sensor 1", "proximity sensor 2",
                              "proximity sensor 3", "proximity sensor 4", "proximity sensor 5", "proximity sensor 6",
                              "proximity sensor 7", "velocity left", "velocity right"]
                    writer = csv.writer(file)

                    if not file_exists:
                        # prints header into file if file doesn't exist yet, otherwise just prints results as row.
                        writer.writerow(header)

                    writer.writerow(self.inputs)
                # Reset
                count = 0
                inputs_avg = []

            else:
                inputs_avg.append(self.inputs)
                count = count + 1

    def sense_compute_and_actuate(self):
        if len(self.inputs) > 0 and len(self.inputsPrevious) > 0:
            # Check for any possible collision
            # because the floor colour of boundary is darker than the center, so we can use ground sensor to detect
            # the boundary and rotate in-place
            if np.max(self.inputs[0:3]) < 0.55:
                self.current_state = self.FSA[5]
            # detect a obstacle
            elif max(self.inputs[11], self.inputs[12], self.inputs[-2], self.inputs[-1]) >= 0.15:
                self.current_state = self.FSA[0]

            elif self.current_state == 1 and (self.inputs[13] > self.distance_max
                                              or self.inputs[-3] > self.distance_max):
                if self.side == "right":
                    if self.inputs[13] > self.distance_max:
                        self.current_state = self.FSA[1]
                        self.distance_max = self.inputs[13]
                elif self.side == "left":
                    if self.inputs[-3] > self.distance_max:
                        self.current_state = self.FSA[1]
                        self.distance_max = self.inputs[-3]
            # when following the obstacle or turning, if the opposite side detect a light source, forward to light source
            elif (self.current_state == 2 or self.current_state == 3) \
                    and ((self.side == "right" and self.inputs[8] == 0)
                         or (self.side == "left" and self.inputs[5] == 0)):
                self.current_state = self.FSA[4]
                self.distance_max = 0.07

            elif (self.current_state == 2 or self.current_state == 3) \
                    and ((self.side == "right" and self.inputs[13] < 0.08)
                         or (self.side == "left" and self.inputs[-3] < 0.08)):
                self.current_state = self.FSA[3]

            elif (self.current_state == 1 or self.current_state == 3) \
                    and (self.inputs[13] < self.distance_max
                         or self.inputs[-3] < self.distance_max):
                if self.side == "right":
                    if self.inputs[13] < self.distance_max:
                        self.current_state = self.FSA[2]
                elif self.side == "left":
                    if self.inputs[-3] < self.distance_max:
                        self.current_state = self.FSA[2]
            # no obstacle, forward to the light source
            elif (self.current_state != 1 and self.current_state != 2) and np.min(self.inputs[3:11]) == 0:
                self.current_state = self.FSA[4]
            # wandering
            elif np.min(self.inputs[3:11]) > 0.8 and np.max(self.inputs[11:19]) < 0.08:
                self.current_state = self.FSA[6]

            if self.current_state == 0:
                pass

            elif self.current_state == 1:
                max_direction = max(self.inputs[11], self.inputs[12], self.inputs[-2], self.inputs[-1])
                if max_direction >= 0.15:
                    if max_direction == max(self.inputs[11], self.inputs[12]):
                        self.side = "right"
                    else:
                        self.side = "left"
                if self.side == "right":
                    self.velocity_left = -self.rotation_speed / 2
                    self.velocity_right = self.rotation_speed / 2
                elif self.side == "left":
                    self.velocity_left = self.rotation_speed / 2
                    self.velocity_right = -self.rotation_speed / 2

            elif self.current_state == 2:
                if self.side == "right":
                    if self.inputs[13] < self.distance_max:
                        self.velocity_left = self.rotation_speed / 1.8
                        self.velocity_right = self.rotation_speed / 2
                    elif self.inputs[13] >= self.distance_max:
                        self.velocity_left = self.rotation_speed / 2
                        self.velocity_right = self.rotation_speed / 1.8
                elif self.side == "left":
                    if self.inputs[-3] < self.distance_max:
                        self.velocity_left = self.rotation_speed / 2
                        self.velocity_right = self.rotation_speed / 1.8
                    elif self.inputs[-3] >= self.distance_max:
                        self.velocity_left = self.rotation_speed / 1.8
                        self.velocity_right = self.rotation_speed / 2

            elif self.current_state == 3:
                if self.side == "right":
                    self.velocity_left = self.rotation_speed / 4
                    self.velocity_right = 0.1
                elif self.side == "left":
                    self.velocity_left = 0.1
                    self.velocity_right = self.rotation_speed / 4

            elif self.current_state == 4:
                # calculate the direction of the light source
                count_left = 0
                count_right = 0
                # count how many sensor detect the light source on the right-hand side
                for i in self.inputs[3:7]:
                    if i == 0:
                        count_right = count_right + 1
                # count how many sensor detect the light source on the left-hand side
                for i in self.inputs[7:11]:
                    if i == 0:
                        count_left = count_left + 1
                # because only one light source at the same time, when two side have same value of detected sensors,
                # two possible situations, one is the light source on the front, another is the light source on the back,
                # so, having to know which case by seeing whether the back light sensor is zero or not.
                if count_right == count_left:
                    if self.inputs[6] == 0 and self.inputs[7] == 0:
                        self.velocity_left = -4
                        self.velocity_right = 4
                    else:
                        self.velocity_left = 4
                        self.velocity_right = 4
                # light source more likely on the right-hand side
                elif count_right > count_left:
                    self.velocity_left = 3
                    self.velocity_right = 1
                # vice versa
                elif count_right < count_left:
                    self.velocity_left = 1
                    self.velocity_right = 3

            elif self.current_state == 5:
                if self.side == "right":
                    self.velocity_left = -4
                    self.velocity_right = 4
                elif self.side == "left":
                    self.velocity_left = 4
                    self.velocity_right = -4

            elif self.current_state == 6:
                self.velocity_left = np.random.rand() * 4
                self.velocity_right = np.random.rand() * 4

        self.left_motor.setVelocity(self.velocity_left)
        self.right_motor.setVelocity(self.velocity_right)

        return [self.velocity_left, self.velocity_right]


if __name__ == "__main__":
    my_robot = Robot()
    controller = Controller(my_robot)
    controller.run_robot("results_task2")
