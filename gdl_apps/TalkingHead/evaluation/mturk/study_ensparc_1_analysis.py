import os, sys 
from pathlib import Path
from glob import glob
import tqdm.auto as tqdm
import random
import yaml
import datetime
import shutil
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt


path_to_result_csv = "/is/cluster/fast/scratch/rdanecek/studies/enspark_v2/results_ensparc_1_pilot.csv"
path_to_protocols = "/is/cluster/fast/scratch/rdanecek/studies/enspark_v2/study_1/"


def analyze_participant(header, participant_results, protocol, discard_repeats=True): 
    # assignment_label = '"AssignmentId"'
    # worker_label = '"WorkerId"'
    # img_label = '"Input.images"'
    # answer_label = '"Answer.1"'
    # answer_hit_label = '"Answer.hit_images"'
    # answer_submit_values_label = '"Answer.submitValues"'
    participant_results = participant_results.split(",")
    
    # assignment_idx = header.index(assignment_label)
    # worker_idx = header.index(worker_label)
    # img_idx = header.index(img_label)
    # answer_idx = header.index(answer_label)
    # answer_hit_idx = header.index(answer_hit_label)
    # answer_submit_values = header.index(answer_submit_values_label)

    # # get the assignment id
    # assignment_id = participant_results[assignment_idx]
    # # get the worker id
    # worker_id = participant_results[worker_idx]
    # # get the images
    # images = participant_results[img_idx]
    # # get the answer
    # answer = participant_results[answer_idx]
    # # get the hit images
    # hit_images = participant_results[answer_hit_idx]
    # # get the submit values
    # submit_values = participant_results[answer_submit_values]

    task_list = participant_results[-2].split(";")
    # TODO: sanity check the task list with the protocol/csv file

    answers = participant_results[-1].strip().strip('"').split(";")
    # filter out empty strings 
    answers = [answer for answer in answers if answer != ""]
    answers_emo = answers[::2]
    answers_lip = answers[1::2]

    num_repeats = protocol["num_repeats"]
    flips = protocol["flips"][0]
    catch_trials = protocol["catch_trials"][0]

    assert len(answers_emo) == len(answers_lip) == len(flips) == len(catch_trials), "Number of answers is different from the expected number of answers."

    model_a = protocol["model_a"]
    model_b = protocol["model_b"]

    flipping = {
        "1": "5", 
        "2": "4",
        "3": "3",
        "4": "2",
        "5": "1"
    }

    unflipped_answers_emo = []
    unflipped_answers_lip = []
    for i in range(len(answers_emo)):
        if flips[i]:
            unflipped_answers_emo += [flipping[answers_emo[i]]]
            unflipped_answers_lip += [flipping[answers_lip[i]]]

    if discard_repeats: 
        unflipped_answers_emo = unflipped_answers_emo[num_repeats:]
        unflipped_answers_lip = unflipped_answers_lip[num_repeats:]
        flips = flips[num_repeats:]
        task_list = task_list[num_repeats:]
        answers = answers[num_repeats:]
        catch_trials = catch_trials[num_repeats:]

    model_preferences_emo = [0] * 5
    catch_preferences_emo = [0] * 5
    model_preferences_lip = [0] * 5
    catch_preferences_lip = [0] * 5
    for i in range(len(unflipped_answers_emo)):
        if catch_trials[i] == 1:
            catch_preferences_emo[int(unflipped_answers_emo[i])-1] += 1
        else:
            model_preferences_emo[int(unflipped_answers_emo[i])-1] += 1

    for i in range(len(unflipped_answers_lip)):
        if catch_trials[i] == 1:
            catch_preferences_lip[int(unflipped_answers_lip[i])-1] += 1
        else:
            model_preferences_lip[int(unflipped_answers_lip[i])-1] += 1
    
    caught_emo = False
    if sum(catch_preferences_emo[:2]) < sum(catch_preferences_emo[2:]):
        caught_emo = True

    caught_lips = False
    if sum(catch_preferences_lip[:2]) < sum(catch_preferences_lip[2:]):
        caught_lips = True


    return np.array(model_preferences_emo), np.array(model_preferences_lip), caught_lips, caught_emo

    
def analyze_single_batch(header, result_lines, assignment_id, protocol):
    assignment_label = '"AssignmentId"'
    assignment_idx = header.index(assignment_label)
    # find all the results with the same assignment id
    assignment_results = [result for result in result_lines if result.split(",")[assignment_idx] == assignment_id]
    # analyze the participant
    participant_results = []
    summed_preferences_emo = np.array([0] * 5)
    summed_preferences_lip = np.array([0] * 5)
    all_preferences_emo = []
    all_preferences_lip = []
    num_participants = len(assignment_results)
    num_useful_participants = len(assignment_results)
    for result in assignment_results:
        preferences_emo, preferences_lip, caught_emo, caught_lips = analyze_participant(header, result, protocol)
        if caught_emo or caught_lips:
            num_useful_participants -= 1
            continue
        all_preferences_emo += [preferences_emo]
        all_preferences_lip += [preferences_lip]
        summed_preferences_emo += preferences_emo
        summed_preferences_lip  += preferences_lip     
    avg_preferences_emo = summed_preferences_emo / num_useful_participants
    avg_preferences_lip = summed_preferences_lip / num_useful_participants

    # plot the results as bar charts, figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    # plot the average preferences
    ax1.bar(["Strongly ours", "Weakly ours", "Indifferent", "Weakly other", "Stongly other"], avg_preferences_emo, color="blue")
    ax1.set_title("Average Preferences for Emotion")
    ax1.set_ylabel("Average Preference")
    ax1.set_xticks(range(0, 5))
    ax1.set_xticklabels(["Strongly\n ours", "Weakly\n ours", "Indifferent", "Weakly\n other", "Stongly\n other"])

    ax2.bar(["Strongly ours", "Weakly ours", "Indifferent", "Weakly other", "Stongly other"], avg_preferences_lip, color="blue")
    ax2.set_title("Average Preferences for Lips")
    ax2.set_ylabel("Average Preference")
    ax2.set_xticks(range(0, 5))
    ax2.set_xticklabels(["Strongly\n ours", "Weakly\n ours", "Indifferent", "Weakly\n other", "Stongly\n other"])



    return avg_preferences_emo, avg_preferences_lip, all_preferences_emo, all_preferences_lip, num_participants, num_useful_participants



def analyze(results, protocol):
    num_participants = len(results)

    # get the header
    header = results[0].split(",")
    result_lines = results[1:]

    assignment_label = '"AssignmentId"'
    assignment_idx = header.index(assignment_label)

    # get all the assignment ids
    assignment_ids = [result.split(",")[assignment_idx] for result in result_lines]
    
    # get the unique assignment ids
    unique_assignment_ids = list(set(assignment_ids))

    batches = {}

    for ai, assignment_id in enumerate(unique_assignment_ids):
        # protocol is an oredered dict, get the protocol by index 
        batch_protocol = protocol[list(protocol.keys())[ai]]

        avg_preferences_emo, avg_preferences_lip, all_preferences_emo, all_preferences_lip, num_participants, num_useful_participants = analyze_single_batch(header, result_lines, assignment_id, batch_protocol)
        batches[ai] = {
            'avg_preferences_emo' : avg_preferences_emo, 
            'avg_preferences_lip' : avg_preferences_lip, 
            'all_preferences_emo' : all_preferences_emo, 
            'all_preferences_lip' : all_preferences_lip, 
            'num_participants' : num_participants, 
            'num_useful_participants' : num_useful_participants
        }
        # print resutls for this batch
        print("Batch: ", ai)
        print("Number of participants: ", num_participants)
        print("Number of useful participants: ", num_useful_participants)
        print("Average preferences for emotion: ", avg_preferences_emo)
        print("Average preferences for lip: ", avg_preferences_lip)        
        print("Preference emo A:", 2*avg_preferences_emo[0] +  avg_preferences_emo[1])
        print("Preference emo B:", 2*avg_preferences_emo[-1] +  avg_preferences_emo[-2])
        print("Preference lip A:", 2*avg_preferences_lip[0] +  avg_preferences_lip[1])
        print("Preference lip B:", 2*avg_preferences_lip[-1] +  avg_preferences_lip[-2])
        
    plt.show()

def main():
    protocol_root = Path(path_to_protocols)
    # find every protocol.yaml in the folder
    protocols = sorted(list(protocol_root.glob("**/protocol.yaml")))
    protocol_dict = OrderedDict()
    for pr in protocols:
        # open the protocol yaml 
        with open(pr, "r") as f:
            protocol = yaml.load(f, Loader=yaml.FullLoader)
        protocol_dict[Path(pr).parent.name] = protocol

    # open the results csv
    with open(path_to_result_csv, "r") as f:
        results = f.readlines()

    analyze(results, protocol_dict)

    print("Done")
    



if __name__ == "__main__":
    main()
