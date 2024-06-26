import torch
from model import NeuralNetwork
import cv2
import numpy as np
import time
import math


def calculate_objective_direction_angle(rocket, target_position):
    # Calculate the difference between the rocket angle and the angle needed to face the target in degrees
    # The angle should be 0 if the rocket is facing the target

    rocket_angle_in_degrees = torch.rad2deg(rocket[2])
    delta_x = target_position[0] - rocket[0]
    delta_y = target_position[1] - rocket[1]
    angle_to_target = torch.atan2(delta_y, delta_x)
    angle_to_target = torch.rad2deg(angle_to_target)
    relative_angle = (angle_to_target -
                      rocket_angle_in_degrees + 180) % 360 - 180

    return relative_angle


def calculate_distance_to_target(rocket_position, target_position):
    # Calculate the distance between the rocket and the target
    delta_x = target_position[0] - rocket_position[0]
    delta_y = target_position[1] - rocket_position[1]
    return torch.sqrt(delta_x ** 2 + delta_y ** 2)


def add_to_rocket_angle(rockets, angle, device):
    new_rockets = rockets.clone()
    angle_tensor = torch.tensor(angle, device=device)
    angle_in_radians = torch.deg2rad(angle_tensor)
    for i, rocket in enumerate(rockets):
        actual_angle = rocket[2]
        new_angle = (actual_angle + angle_in_radians) % (2 * 3.1415)
        new_rockets[i, 2] = new_angle
    return new_rockets


def calculate_next_position(rockets, device):
    new_rockets = rockets.clone()
    positions = []
    angles = []
    objective_directions = []
    objective_distances = []
    for i, rocket in enumerate(rockets):
        new_x = rocket[0] + rocket[3] * torch.cos(new_rockets[i, 2])
        new_y = rocket[1] + rocket[3] * torch.sin(new_rockets[i, 2])
        new_angle = calculate_objective_direction_angle(torch.tensor(
            [new_x, new_y, rocket[2]], device=device), torch.tensor([150, 100], device=device))
        positions.append([new_x, new_y])
        angles.append(rocket[2])
        objective_directions.append(new_angle)
        objective_distances.append(
            calculate_distance_to_target(torch.tensor([new_x, new_y, new_angle], device=device), torch.tensor([150, 100], device=device)))

    new_rockets[:, :2] = torch.tensor(positions, device=device)
    new_rockets[:, 2] = torch.tensor(angles, device=device)
    new_rockets[:, 4] = torch.tensor(objective_directions, device=device)
    new_rockets[:, 5] = torch.tensor(objective_distances, device=device)

    return new_rockets


def calculate_desired_output(rocket, target):
    # Calculate the angle to the target in degrees
    angle_to_target = calculate_objective_direction_angle(rocket[:3], target)
    # Initialize the desired output as a 3-element vector
    desired_output = torch.zeros(3)

    # If the angle to the target is positive, the rocket should turn right
    if angle_to_target >= 0:
        desired_output[1] = 1
    # If the angle to the target is negative, the rocket should turn left
    elif angle_to_target < 0:
        desired_output[0] = 1
    # If the angle to the target is 0, the rocket should do nothing
    else:
        desired_output[2] = 1

    return desired_output


def draw_and_watch_rocket(rockets):
    # Create a black canvas
    canvas = np.zeros((500, 500, 3), dtype=np.uint8)
    # Draw the target as a green circle
    target_position = (150, 100)
    for rocket in rockets:
        # Draw the rocket as a red circle
        rocket_position = (int(rocket[0].item()), int(rocket[1].item()))
        cv2.circle(canvas, rocket_position, 2, (0, 0, 255), -1)
        # Draw the rocket's direction as a line
        direction_end = (
            int(rocket_position[0] + 10 * np.cos(rocket[2].item())),
            int(rocket_position[1] + 10 * np.sin(rocket[2].item()))
        )
        cv2.line(canvas, rocket_position, direction_end, (0, 0, 255), 1)

    cv2.circle(canvas, target_position, 5, (0, 255, 0), -1)

    # Show the canvas
    cv2.imshow("Rocket", canvas)
    cv2.waitKey(1)


def main():
    print("Starting the program...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    model = NeuralNetwork().to(device)

    ROCKET_NUMBERS = 20

    ROCKET_POSX_RANGE = [200, 300]
    ROCKET_POSY_RANGE = [390, 410]
    ROCKET_ANGLE_RANGE = [0, 180]

    ROCKET_VELOCITY = 2

    # Create target rocket
    target = torch.tensor([150, 100], dtype=torch.float32).to(device)

    # Create the criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(100):
        count = 0
        ROCKET_POSX = torch.randint(
            ROCKET_POSX_RANGE[0], ROCKET_POSX_RANGE[1], (ROCKET_NUMBERS,))
        ROCKET_POSY = torch.randint(
            ROCKET_POSY_RANGE[0], ROCKET_POSY_RANGE[1], (ROCKET_NUMBERS,))
        ROCKET_ANGLE = torch.tensor([math.radians(angle) for angle in torch.randint(
            ROCKET_ANGLE_RANGE[0], ROCKET_ANGLE_RANGE[1], (ROCKET_NUMBERS,))])
        # Set initial position of the rockets to the initial position of the train_rockets
        for i in range(ROCKET_NUMBERS):
            initial_objective_direction = calculate_objective_direction_angle(
                torch.tensor([ROCKET_POSX[i], ROCKET_POSY[i], ROCKET_ANGLE[i]], dtype=torch.float32).to(device), target)
            initial_objective_distance = calculate_distance_to_target(
                torch.tensor([ROCKET_POSX[i], ROCKET_POSY[i], ROCKET_ANGLE[i]], dtype=torch.float32).to(device), target)
            train_rocket = torch.tensor(
                [[ROCKET_POSX[i], ROCKET_POSY[i], ROCKET_ANGLE[i], ROCKET_VELOCITY, initial_objective_direction, initial_objective_distance]], dtype=torch.float32, ).to(device)
            test_rocket = torch.tensor(
                [[ROCKET_POSX[i], ROCKET_POSY[i], ROCKET_ANGLE[i], ROCKET_VELOCITY, initial_objective_direction, initial_objective_distance]], dtype=torch.float32, requires_grad=True).to(device)
            # Append train_rocket to train_rockets
            if i == 0:
                train_rockets = train_rocket
                test_rockets = test_rocket
            else:
                train_rockets = torch.cat((train_rockets, train_rocket), 0)
                test_rockets = torch.cat((test_rockets, test_rocket), 0)

        rockets = train_rockets.clone()
        # Set rockets position to the initial position

        while count < 250:
            # Execute the model
            # output[0] = turn left, output[1] = turn right, output[2] = do nothing
            outputs = model(rockets)
            for i, output in enumerate(outputs):
                if output[0] > output[1] and output[0] > output[2]:
                    new_rockets = add_to_rocket_angle(
                        rockets[i].unsqueeze(0), -3, device)
                    rockets = rockets.clone()
                    rockets[i, 2] = new_rockets[0, 2]
                elif output[1] > output[0] and output[1] > output[2]:
                    new_rockets = add_to_rocket_angle(
                        rockets[i].unsqueeze(0), 3, device)
                    rockets = rockets.clone()
                    rockets[i, 2] = new_rockets[0, 2]
                elif output[2] > output[0] and output[2] > output[1]:
                    pass
            # If rockets are out of the canvas, stop the simulation and reset the rocket
            # new_rockets = rockets.clone()

            # for i, rocket in enumerate(rockets):
            #     if rocket[0] < 0 or rocket[0] > 501 or rocket[1] < 0 or rocket[1] > 500:
            #         new_rockets[i, 3] = 0
            # rockets = new_rockets
           # If rockets are in the target set velocity to 0
            new_rockets = rockets.clone()
            for i, rocket in enumerate(rockets):
                if rocket[5] < 5:
                    new_rockets[i, 3] = 0
            rockets = new_rockets
            # Calculate the desired direction for each rocket
            draw_and_watch_rocket(rockets)
            desired_outputs = torch.stack(
                [calculate_desired_output(rocket, target) for rocket in rockets]).to(device)
            # Calculate the new position
            rockets = calculate_next_position(rockets, device)
            # Calculate the loss
            loss = criterion(outputs, desired_outputs)
            # Zero the gradients
            optimizer.zero_grad()
            # Backward pass
            loss.backward(retain_graph=True)
            # Optimize
            weights_before = model.state_dict()
            optimizer.step()
            weights_after = model.state_dict()
            count += 1
            print(count)
            # time.sleep(1)
            print(f"Epoch: {epoch}, Loss: {loss.item()}")
        print(f"Epoch: {epoch}, Loss: {loss.item()}")


main()
