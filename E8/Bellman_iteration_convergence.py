import numpy as np
import matplotlib.pyplot as plt

#r value to be changed
r = -3

#discount factor
dis_fac = 0.5


#iterations to train
it = 20

#create numpy matrix for states
utilities_states = np.array([[r, -1, 10],[-1, -1, -1],[-1, -1, -1]])


#create lists for state values histories
state_history = []
for state in utilities_states.flatten():
    state_history.append([])



def max_Sum(state_row, state_collum):
    scan = [-1,1]
    sum_list = []
    #calculate sum for all four possibilties
    for direction in scan:
        first_sum = utilities_states[state_row][state_collum+direction] * 0.8 if (state_collum + direction < 3 and state_collum + direction >= 0 ) else utilities_states[state_row][state_collum] * 0.8
        first_sum += utilities_states[state_row+direction][state_collum] * 0.1 if (state_row + direction < 3 and state_row + direction >= 0 ) else utilities_states[state_row][state_collum] * 0.1
        first_sum += utilities_states[state_row-direction][state_collum] * 0.1 if (state_row - direction < 3 and state_row - direction >= 0 ) else utilities_states[state_row][state_collum] * 0.1     

        second_sum = utilities_states[state_row+direction][state_collum] * 0.8 if (state_row + direction < 3 and state_row + direction >= 0 ) else utilities_states[state_row][state_collum] * 0.8
        second_sum += utilities_states[state_row][state_collum+direction] * 0.1 if (state_collum + direction < 3 and state_collum + direction >= 0 ) else utilities_states[state_row][state_collum] * 0.1
        second_sum += utilities_states[state_row][state_collum-direction] * 0.1 if (state_collum - direction < 3 and state_collum - direction >= 0 ) else utilities_states[state_row][state_collum] * 0.1

        sum_list.append(first_sum)
        sum_list.append(second_sum)

    #get max value from sum list
    max_sum = max(sum_list)

    return max_sum


def train():
    for _ in range(it):
        for row in range(3):
            for collum in range(3):
                #Add to respective history
                index = row * 3 + collum
                state_history[index].append(utilities_states[row][collum])

                #define penalty
                pen = 10 if(row==0 and collum == 2) else -1

                #Calculate new Utility
                utilities_states[row][collum] = pen + dis_fac * max_Sum(row,collum)

    

train()


# Plotting the state value histories
for i, state_values in enumerate(state_history):
    row = i // 3
    col = i % 3
    plt.plot(state_values, label=f"State ({row}, {col})")

plt.xlabel("Iterations")
plt.ylabel("State Value")
plt.legend()
plt.show()