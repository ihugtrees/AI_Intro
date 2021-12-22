import numpy as np
import math
from grid import Grid


def value_change(value_array, value_array_2):
    """
    :param value_array: first value array
    :param value_array_2: second value array
    :return: The maximum error between two value iterations
    """
    max_change = 0
    for state, value in value_array.items():
        change = abs(value - value_array_2[state])
        max_change = max(change, max_change)
    return max_change


def value_iteration_update(instance, all_states, q_table, previous_iteration_value_array,
                           current_iteration_value_array):
    """
    :param instance: Instance object of the problem we are solving
    :param all_states: List of all valid action states
    :param q_table: nested dictionary in the form (x,y): utility for each state, i.e (1,1):{'south':0.1, 'north':0.9 ...}
    :param previous_iteration_value_array: previous iteration dictionary in the form (x, y): value for each valid state
    :param current_iteration_value_array: current (new value array for each iteration) dictionary in the form (x, y): value for each valid state
    :return: previous_iteration_value_array, current_iteration_value_array (after update), q_table (after update)
    """
    # TODO: implement value iteration update function
    # print("Value iteration update function is not implemented yet")

    for state in all_states:
        for action, probs in instance.get_actions(state).items():
            reward = instance.get_reward(state, action)
            nbors_sum = 0
            for nbor, prob in probs.items():
                nbors_sum += prob * previous_iteration_value_array[nbor]
            q_table[state][action] = reward + nbors_sum
        current_iteration_value_array[state] = max(q_table[state].values())
    return previous_iteration_value_array, current_iteration_value_array, q_table


def get_policy(instance, q_table):
    """
    :param instance: instance of the problem
    :param q_table: latest q_table
    :return: Dictionary of policy in the form: (x,y): 'direction'
    """
    policy = instance.create_policy_array()
    for state in q_table.keys():  # For each s in S
        # TODO: create policy based on the latest q_table
        # print("get policy for value iteration is not implemented yet")
        policy[state] = max(q_table[state], key=q_table[state].get)

    # demo_policy = {(0, 0): 'north', (0, 1): 'north', (1, 1): 'south'}
    # print("Policy for exmaple ", demo_policy)
    return policy


def value_iteration(instance):
    """
    :param instance: Grid class object that holds the instance of the problem
    :return: policy - matrix that holds which direction to go in each cell
    """
    epsilon = 0.001
    previous_iteration_value_array = instance.create_value_array()  # dic of state:utility, holds the previous iteration
    current_iteration_value_array = instance.create_value_array()  # same as above, holds the current iteration value
    q_table = instance.create_q_values_table()  # q table that holds the utility for each state,action
    all_states = instance.get_all_states()  # List of all valid actions states
    policy = instance.create_policy_array()  # starting policy, all states direct "south"
    c = 0
    while True:
        previous_iteration_value_array, current_iteration_value_array, q_table = \
            value_iteration_update(instance, all_states, q_table, previous_iteration_value_array,
                                   current_iteration_value_array)
        # TODO: Condition to stop the iterations, add the relevant function here
        if value_change(previous_iteration_value_array, current_iteration_value_array) < epsilon:
            # print('Needs to implement the stop condition for the loop')
            print(f'finished after {c} iterations')
            break
        previous_iteration_value_array = current_iteration_value_array
        current_iteration_value_array = instance.create_value_array()
        c += 1
    return get_policy(instance, q_table)
