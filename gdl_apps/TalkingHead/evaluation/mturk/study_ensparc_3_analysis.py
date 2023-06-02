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
import seaborn as sns


# path_to_result_csv = "/is/cluster/fast/scratch/rdanecek/studies/enspark_v2/results_ensparc_3_pilot.csv"
# path_to_protocols = "/is/cluster/fast/scratch/rdanecek/studies/enspark_v2/study_3/"


path_to_result_csv = "/is/cluster/fast/scratch/rdanecek/studies/enspark_final_v1/ENSPARC_3_main_study.csv"
path_to_protocols = "/is/cluster/fast/scratch/rdanecek/studies/enspark_final_v1/study_3/"


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

    task_str = participant_results[-2]
    task_list = task_str.split(";")
    # TODO: sanity check the task list with the protocol/csv file

    answers = participant_results[-1].strip().strip('"').split(";")
    # filter out empty strings 
    answers = [answer for answer in answers if answer != ""]
    answers_lip = answers

    num_repeats = protocol["num_repeats"]
    flips = protocol["flips"][0]
    catch_trials = protocol["catch_trials"][0]

    # assert len(answers_lip) == len(flips) == len(catch_trials), "Number of answers is different from the expected number of answers."
    
    if not len(answers_lip) == len(flips) == len(catch_trials):
        answers_lip = answers_lip[:len(flips)]
        # # filter out by wrong number of answers
        # print("[WARNING]: Number of answers is different from the expected number of answers. Skipping participant")
        # return None, False

    model_a = protocol["model_a"]
    model_b = protocol["model_b"]

    sanity_passed = False
    for task in task_list: 
        vid_a, vid_b = task.split("#")
        vid_a = vid_a.strip('"')
        vid_b = vid_b.strip('"')
        if (model_a in vid_a or model_a in vid_b) and   (model_b in vid_a or model_b in vid_b):
            sanity_passed = True
            break
    assert sanity_passed, "The participant did not see the expected videos."

    flipping = {
        "1": "5", 
        "2": "4",
        "3": "3",
        "4": "2",
        "5": "1"
    }

    
    unflipped_answers_lip = []
    for i in range(len(answers_lip)):
        if flips[i]:
            unflipped_answers_lip += [flipping[answers_lip[i]]]

    if discard_repeats: 
        unflipped_answers_lip = unflipped_answers_lip[num_repeats:]
        flips = flips[num_repeats:]
        task_list = task_list[num_repeats:]
        answers = answers[num_repeats:]
        catch_trials = catch_trials[num_repeats:]

    model_preferences_lip = [0] * 5
    catch_preferences_lip = [0] * 5

    for i in range(len(unflipped_answers_lip)):
        if catch_trials[i] == 1:
            catch_preferences_lip[int(unflipped_answers_lip[i])-1] += 1
        else:
            model_preferences_lip[int(unflipped_answers_lip[i])-1] += 1
    
    caught_lips = False
    if sum(catch_preferences_lip[:2]) < sum(catch_preferences_lip[2:]):
        caught_lips = True

    return  np.array(model_preferences_lip), caught_lips

    
def analyze_single_batch(header, result_lines, protocol):
    # assignment_label = '"AssignmentId"'
    # assignment_idx = header.index(assignment_label)
    # find all the results with the same assignment id
    # assignment_results = [result for result in result_lines if result.split(",")[assignment_idx] == assignment_id]
    assignment_results = result_lines
    # analyze the participant
    participant_results = []
    summed_preferences_lip = np.array([0] * 5)
    all_preferences_lip = []
    num_participants = len(assignment_results)
    num_useful_participants = len(assignment_results)
    for ri, result in enumerate(assignment_results):
        preferences_lip, caught_lips = analyze_participant(header, result, protocol)
        if caught_lips:
            num_useful_participants -= 1
            continue
        all_preferences_lip += [preferences_lip]
        summed_preferences_lip  += preferences_lip     
    avg_preferences_lip = summed_preferences_lip / num_useful_participants

    return avg_preferences_lip, all_preferences_lip, num_participants, num_useful_participants



def get_batch_results(results, protocol):
    model_b = protocol["model_b"]
    indices = []
    batch_results = []
    for ri in range(len(results)):
        task_str = results[ri]
        task_list = task_str.split(",")[-2].strip('"').split(";")
        for ti, task in enumerate(task_list): 
            if model_b in task:
                indices += [ri]
                batch_results += [results[ri]]
                break
    return batch_results




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


    num_batches = len(protocol)
    num_participants = len(result_lines) // num_batches
    assert len(result_lines) % num_batches == 0, "Number of participants is not divisible by number of batches."
    for batch_i, (protocol_name, batch_protocol) in enumerate (protocol.items()):
        # protocol is an oredered dict, get the protocol by index 
        
        # num_participants = batch_protocol["num_participants"]

        # batch_results = result_lines[batch_i * num_participants : (batch_i + 1) * num_participants]
        batch_results = get_batch_results(result_lines, batch_protocol)

        avg_preferences_lip, all_preferences_lip, num_participants, num_useful_participants \
            = analyze_single_batch(header, batch_results, batch_protocol)
        std_preferences_lip = np.std(np.stack( all_preferences_lip, axis=0), axis=0)
        batches[batch_i] = {
            'protocol_name' : protocol_name,
            'model_a' : batch_protocol["model_a"],
            'model_b' : batch_protocol["model_b"],
            'avg_preferences_lip' : avg_preferences_lip, 
            'std_preferences_lip' : std_preferences_lip, 
            'all_preferences_lip' : all_preferences_lip, 
            'num_participants' : num_participants, 
            'num_useful_participants' : num_useful_participants
        }
        # print resutls for this batch
        print("Batch: ", batch_i)
        print("Model A: ", batch_protocol["model_a"])
        print("Model B: ", batch_protocol["model_b"])
        print("Number of participants: ", num_participants)
        print("Number of useful participants: ", num_useful_participants)
        print("Average preferences for lip sync: ", avg_preferences_lip)
        print("Preference A:", 2*avg_preferences_lip[0] +  avg_preferences_lip[1])
        print("Preference B:", 2*avg_preferences_lip[-1] +  avg_preferences_lip[-2])

    #     fig = plt.figure()
    #     ax = fig.add_subplot(111)
    #     ax.bar(["Strongly ours", "Weakly ours", "Indifferent", "Weakly other", "Stongly other"], avg_preferences_lip, color="blue")
    #     ax.set_title(f"Average Preferences for Lips: Ours vs {batch_protocol['model_b']}")
    #     ax.set_ylabel("Average Preference")
    #     ax.set_xticks(range(0, 5))
    #     ax.set_xticklabels(["Strongly\n ours", "Weakly\n ours", "Indifferent", "Weakly\n other", "Stongly\n other"])

    # plt.show()
    order = [3, 0, 2, 1, 4]

    merged_hist(batches, order)


model_name_dict= {
    'CT_test_reorg' : 'CodeTalker',
    'VOCA_test_reorg' : 'VOCA',
    'FF_test_reorg' : 'FaceFormer',
    'MT_test_reorg' : 'MeshTalk',
    '2023_05_10_13-16-00_-3885098104460673227_FaceFormer_MEADP_Awav2vec2T_Elinear_DFaceFormerDecoder_Seml_PPE_predV_LV' : 'FaceFormer-MEAD',
}

def merged_hist(batches, order=None, with_std=False, fig=None):
    # Suppose we have n models and their results
    n = len(batches)  # replace with your number of models
    order = list(range(n)) if order is None else order
    # Define the number of bins and the bar width
    bins = np.linspace(-4, 4, 5)  # change bin numbers as needed
    width = (bins[1] - bins[0]) / (n + 1)  # adjust for the number of models

    sns.set_style("whitegrid")
    colors = sns.color_palette("Set2", n)

    if fig is None:
        fig = plt.figure(figsize=(8,6))

    # Plot the results of each model
    # for i, batch in enumerate(batches):
    for i, order_idx in enumerate(order):
        result = batches[order_idx]["avg_preferences_lip"]
        std = batches[order_idx]["std_preferences_lip"]
        x_coords = bins - width*(n/2) + i*width
        plt.bar(x_coords, result, width=width, 
                label=f'Ours vs  {model_name_dict[ batches[order_idx]["model_b"]]}', color=colors[order_idx])
        if with_std:
            plt.errorbar(x_coords, result, yerr=std, fmt='none', color='k')

    # Add legend and labels
    plt.legend()
    # plt.xlabel('Value')
    plt.ylabel('Participant average preference for lip sync')
    plt.title("Average preferences for lip sync: EMOTE vs SOTA methods")
    fig.gca().set_xticklabels(["", "Strongly\n ours", "Weakly\n ours", "Indifferent", "Weakly\n other", "Stongly\n other"])
    # disable x ticks but keep the labels 
    plt.tick_params(axis='x', which='both', length=0)
    plt.show()
    # export to pdf, no white borders
    fig.savefig(Path(__file__).parent / "study_result_3.pdf", bbox_inches='tight', pad_inches=0)


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
