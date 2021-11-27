from controller import Robot, Receiver, Emitter
import sys, struct, math
import numpy as np
import mlp as ntw
import csv
import os.path

class Controller:
    def __init__(self, robot):
        # Robot Parameters
        # Please, do not change these parameters
        self.robot = robot
        self.time_step = 32  # ms
        self.max_speed = 1  # m/s

        # MLP Parameters and Variables   
        ### Define bellow the architecture of your MLP including the number of neurons on your input, hidden and output layers.
        self.number_input_layer = 9
        self.number_hidden_layer_1 = 12
        self.number_hidden_layer_2 = 10
        self.number_output_layer = 2

        # Initialize the network
        self.network = ntw.MLP(self.number_input_layer, self.number_hidden_layer_1, self.number_hidden_layer_2,
                               self.number_output_layer)
        # Example with 2 hidden layers        #ntw.MLP(self.number_input_layer,self.number_hidden_layer_1,self.number_hidden_layer_2,self.number_output_layer)
        self.inputs = []

        # Calculate the number of weights of your MLP
        self.number_weights = (self.number_input_layer + 1) * self.number_hidden_layer_1 + self.number_hidden_layer_1 * self.number_hidden_layer_2 + self.number_hidden_layer_2 * self.number_output_layer

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

        # Enable Emitter and Receiver (to communicate with the Supervisor)
        self.emitter = self.robot.getDevice("emitter")
        self.receiver = self.robot.getDevice("receiver")
        self.receiver.enable(self.time_step)
        self.receivedData = ""
        self.receivedDataPrevious = ""
        self.flagMessage = False

        # Fitness value (initialization fitness parameters once)
        self.fitness_values = []
        self.fitness = 0

    def clip_value(self, value, min_max):
        if value > min_max:
            return min_max
        elif value < -min_max:
            return -min_max
        return value

    def run_robot(self):
        # Main Loop
        filename = "results_task3_sensor.csv"
        while self.robot.step(self.time_step) != -1:
            # This is used to store the current input data from the sensors
            self.inputs = []

            # Emitter and Receiver
            # Check if there are messages to be sent or read to/from our Supervisor
            self.handle_emitter()
            self.handle_receiver()

            # Read Ground Sensors
            left = self.left_ir.getValue()
            center = self.center_ir.getValue()
            right = self.right_ir.getValue()
            # print("Ground Sensors \n    left {} center {} right {}".format(left, center, right))

            ### Please adjust the ground sensors values to facilitate learning
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

            # Normalize the values between 0 and 1 and save data
            self.inputs.append((left - min_gs) / (max_gs - min_gs))
            self.inputs.append((center - min_gs) / (max_gs - min_gs))
            self.inputs.append((right - min_gs) / (max_gs - min_gs))
            # print("Ground Sensors \n    left {} center {} right {}".format(self.inputs[0],self.inputs[1],self.inputs[2]))

            # Read Distance Sensors
            for i in range(8):
                ### Select the distance sensors that you will use
                if i == 0 or i == 1 or i == 2 or i == 5 or i == 6 or i == 7:
                    temp = self.proximity_sensors[i].getValue()

                    ### Please adjust the distance sensors values to facilitate learning
                    min_ds = 50
                    max_ds = 500

                    if temp > max_ds:
                        temp = max_ds
                    if temp < min_ds:
                        temp = min_ds

                    # Normalize the values between 0 and 1 and save data
                    self.inputs.append((temp - min_ds) / (max_ds - min_ds))
                    # print("Distance Sensors - Index: {}  Value: {}".format(i,self.proximity_sensors[i].getValue()))

            # GA Iteration
            # Verify if there is a new genotype to be used that was sent from Supervisor
            self.check_for_new_genes()
            # Define the robot's actuation (motor values) based on the output of the MLP
            tempV = self.sense_compute_and_actuate()
            # Calculate the fitness value of the current iteration
            tempF = self.calculate_fitness()
            self.inputs = np.hstack((self.inputs, tempV))
            self.inputs = np.hstack((self.inputs, tempF))
            file_exists = os.path.isfile(str(filename))
            with open("results_task3_sensor.csv", "a", newline='') as file:
                header = ["Ground sensor left", "Ground sensor middle", "Ground sensor right", "proximity sensor 0",
                          "proximity sensor 1", "proximity sensor 2", "proximity sensor 5", "proximity sensor 6",
                          "proximity sensor 7", "velocity left", "velocity right", "forward fitness",
                          "follow line fitness", " avoid collision fitness", " spinning fitness", "combined fitness"]
                writer = csv.writer(file)

                if not file_exists:
                    # prints header into file if file doesn't exist yet, otherwise just prints results as row.
                    writer.writerow(header)

                writer.writerow(self.inputs)
            # End of the iteration

    def handle_emitter(self):
        # Send the self.weights value to the supervisor
        data = str(self.number_weights)
        data = "weights: " + data
        string_message = str(data)
        string_message = string_message.encode("utf-8")
        # print("Robot send:", string_message)
        self.emitter.send(string_message)

        # Send the self.fitness value to the supervisor
        data = str(self.fitness)
        data = "fitness: " + data
        string_message = str(data)
        string_message = string_message.encode("utf-8")
        print("Robot send fitness:", string_message)
        self.emitter.send(string_message)

    def handle_receiver(self):

        if self.receiver.getQueueLength() > 0:
            # print("queue length", self.receiver.getQueueLength())
            while self.receiver.getQueueLength() > 0:
                # Adjust the Data to our model
                self.receivedData = self.receiver.getData().decode("utf-8")
                self.receivedData = self.receivedData[1:-1]
                self.receivedData = self.receivedData.split()
                x = np.array(self.receivedData)
                self.receivedData = x.astype(float)
                # print("Controller handle receiver data:", self.receivedData)
                self.receiver.nextPacket()

            # Is it a new Genotype?
            if np.array_equal(self.receivedDataPrevious, self.receivedData) == False:
                self.flagMessage = True

            else:
                self.flagMessage = False

            self.receivedDataPrevious = self.receivedData
        else:
            # print("Controller receiver q is empty")
            self.flagMessage = False

    def check_for_new_genes(self):
        if self.flagMessage == True and self.receivedData != "":
            # Receive genotype and set the weights of the network
            # print("\n New genotype")
            self.data = []
            part1 = (self.number_input_layer + 1) * self.number_hidden_layer_1
            part2 = self.number_hidden_layer_1 * self.number_hidden_layer_2
            part3 = self.number_hidden_layer_2 * self.number_output_layer

            self.network.weightsPart1 = self.receivedData[0:part1]
            self.network.weightsPart2 = self.receivedData[part1:part2 + part1]
            self.network.weightsPart3 = self.receivedData[part2 + part1:part3 + part2 + part1]

            self.network.weightsPart1 = self.network.weightsPart1.reshape(
                [self.number_input_layer + 1, self.number_hidden_layer_1])
            self.network.weightsPart2 = self.network.weightsPart2.reshape(
                [self.number_hidden_layer_1, self.number_hidden_layer_2])
            self.network.weightsPart3 = self.network.weightsPart3.reshape(
                [self.number_hidden_layer_2, self.number_output_layer])

            self.data.append(self.network.weightsPart1)
            self.data.append(self.network.weightsPart2)
            self.data.append(self.network.weightsPart3)
            self.network.weights = self.data

            self.fitness_values = []

    def sense_compute_and_actuate(self):
        # MLP:
        #   Input == sensory data
        #   Output == motors commands
        output = self.network.propagate_forward(self.inputs)
        self.velocity_left = output[0] * 3
        self.velocity_right = output[1] * 3

        # Multiply the motor values by 3 to increase the velocities
        self.left_motor.setVelocity(self.velocity_left)
        self.right_motor.setVelocity(self.velocity_right)

        return [self.velocity_left, self.velocity_right]

    def calculate_fitness(self):
        ### Define the fitness function to increase the speed of the robot and
        ### to encourage the robot to move forward only
        # base value is the 7 times of the sum of two speed
        forwardFitness = (self.velocity_left + self.velocity_right) * 7
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
        if np.max(self.inputs[3:9]) >= 0.65:
            avoidCollisionFitness = 0.001
        # when detect a obstacle on the right-hand side, turn left significantly
        elif (0.0622 < self.inputs[3] < 0.65 or 0.0622 < self.inputs[4] < 0.65) and \
                (self.velocity_right > self.velocity_left) and (self.velocity_right - self.velocity_left >= 1.5):
            avoidCollisionFitness = 1.3
        # when detect a obstacle on the left-hand side, turn right significantly
        elif (0.0622 < self.inputs[7] < 0.65 or 0.0622 < self.inputs[8] < 0.65) and \
                (self.velocity_right < self.velocity_left) and (self.velocity_left - self.velocity_right >= 1.5):
            avoidCollisionFitness = 1.3
        # when parallel with the obstacle, moving forward with slight curve
        elif (0.2 < self.inputs[5] < 0.65 or 0.2 < self.inputs[6] < 0.65) and \
                abs(self.velocity_right - self.velocity_left) < 1.5:
            avoidCollisionFitness = 1.3
        # no obstacle case
        elif np.max(self.inputs[3:9]) < 0.0667:
            avoidCollisionFitness = 1
        else:
            avoidCollisionFitness = 0.5

        ### Define the fitness function to avoid spining behaviour
        # simply can't have a negative speed
        if self.velocity_left < 0 or self.velocity_right < 0:
            spinningFitness = 0
        else:
            spinningFitness = 100
        print("forward fitness {0} ","follow line fit {1}", " avoid collision fit {2}", " spinning fit {3} ",{forwardFitness},{followLineFitness},{avoidCollisionFitness},{spinningFitness})
        ### Define the fitness function of this iteration which should be a combination of the previous functions
        # following the line has highest priority, and then is the speed, then the spinning
        combinedFitness = (forwardFitness * 0.5 + followLineFitness * 0.6 + spinningFitness * 0.4) * avoidCollisionFitness

        self.fitness_values.append(combinedFitness)
        self.fitness = np.mean(self.fitness_values)

        return [forwardFitness, followLineFitness, avoidCollisionFitness, spinningFitness, combinedFitness]


if __name__ == "__main__":
    # Call Robot function to initialize the robot
    my_robot = Robot()
    # Initialize the parameters of the controller by sending my_robot
    controller = Controller(my_robot)
    # Run the controller
    controller.run_robot()
