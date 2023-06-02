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


# path_to_result_csv = "/is/cluster/fast/scratch/rdanecek/studies/enspark_v2/results_ensparc_1_pilot.csv"
# path_to_protocols = "/is/cluster/fast/scratch/rdanecek/studies/enspark_v2/study_1/"


path_to_result_csv = "/is/cluster/fast/scratch/rdanecek/studies/enspark_final_v1/ENSPARC_1_main_study.csv"
path_to_protocols = "/is/cluster/fast/scratch/rdanecek/studies/enspark_final_v1/study_1/"



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

    num_repeats = protocol["num_repeats"]
    flips = protocol["flips"][0]
    catch_trials = protocol["catch_trials"][0]

    answers = participant_results[-1].strip().strip('"').split(";")
    # filter out empty strings 
    answers = [answer for answer in answers if answer != ""]
    answers_emo = answers[::2]
    if len(answers_emo) >  len(flips):
        answers_emo = answers_emo[:len(flips)]
    answers_lip = answers[1::2]
    if len(answers_lip) >  len(flips):
        answers_lip = answers_lip[:len(flips)]


    assert len(answers_emo) == len(answers_lip) == len(flips) == len(catch_trials), \
        f"Number of answers is different from the expected number of answers the lengths are" \
            f"{len(answers_emo)}, {len(answers_lip)}, {len(flips)}, {len(catch_trials)}"

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

    
# def analyze_single_batch(header, result_lines, assignment_id, protocol):
def analyze_single_batch(header, result_lines, protocol):
    # assignment_label = '"AssignmentId"'
    # assignment_idx = header.index(assignment_label)
    # find all the results with the same assignment id
    # assignment_results = [result for result in result_lines if result.split(",")[assignment_idx] == assignment_id]
    assignment_results = result_lines
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

    # # plot the results as bar charts, figure with two subplots
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    # # plot the average preferences
    # ax1.bar(["Strongly ours", "Weakly ours", "Indifferent", "Weakly other", "Stongly other"], avg_preferences_emo, color="blue")
    # ax1.set_title("Average Preferences for Emotion")
    # ax1.set_ylabel("Average Preference")
    # ax1.set_xticks(range(0, 5))
    # ax1.set_xticklabels(["Strongly\n ours", "Weakly\n ours", "Indifferent", "Weakly\n other", "Stongly\n other"])

    # ax2.bar(["Strongly ours", "Weakly ours", "Indifferent", "Weakly other", "Stongly other"], avg_preferences_lip, color="blue")
    # ax2.set_title("Average Preferences for Lips")
    # ax2.set_ylabel("Average Preference")
    # ax2.set_xticks(range(0, 5))
    # ax2.set_xticklabels(["Strongly\n ours", "Weakly\n ours", "Indifferent", "Weakly\n other", "Stongly\n other"])



    return avg_preferences_emo, avg_preferences_lip, all_preferences_emo, all_preferences_lip, num_participants, num_useful_participants




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
    
    batches = {}


    num_batches = len(protocol)
    num_participants = len(result_lines) // num_batches
    assert len(result_lines) % num_batches == 0, "Number of participants is not divisible by number of batches."
    for batch_i, (protocol_name, batch_protocol) in enumerate (protocol.items()):
        # protocol is an oredered dict, get the protocol by index 
        batch_protocol = protocol[list(protocol.keys())[batch_i]]

        # batch_results = result_lines[batch_i * num_participants : (batch_i + 1) * num_participants]
        # batch_results = result_lines[batch_i * num_participants : (batch_i + 1) * num_participants]
        batch_results = get_batch_results(result_lines, batch_protocol)

        avg_preferences_emo, avg_preferences_lip, all_preferences_emo,\
              all_preferences_lip, num_participants, num_useful_participants \
                = analyze_single_batch(header, batch_results, batch_protocol)
        batches[batch_i] = {
            'protocol_name' : protocol_name,
            'model_a' : batch_protocol['model_a'],
            'model_b' : batch_protocol['model_b'],
            'avg_preferences_emo' : avg_preferences_emo, 
            'avg_preferences_lip' : avg_preferences_lip, 
            'all_preferences_emo' : all_preferences_emo, 
            'all_preferences_lip' : all_preferences_lip, 
            'num_participants' : num_participants, 
            'num_useful_participants' : num_useful_participants
        }
        # print resutls for this batch
        print("Batch: ", batch_i)
        print("Number of participants: ", num_participants)
        print("Number of useful participants: ", num_useful_participants)
        print("Average preferences for emotion: ", avg_preferences_emo)
        print("Average preferences for lip: ", avg_preferences_lip)        
        print("Preference emo A:", 2*avg_preferences_emo[0] +  avg_preferences_emo[1])
        print("Preference emo B:", 2*avg_preferences_emo[-1] +  avg_preferences_emo[-2])
        print("Preference lip A:", 2*avg_preferences_lip[0] +  avg_preferences_lip[1])
        print("Preference lip B:", 2*avg_preferences_lip[-1] +  avg_preferences_lip[-2])
        

    merged_hist(batches, title="Average preferences for emotion: EMOTE vs ablation models",
                output_path=Path(__file__).parent / "study_result_1_emo.pdf",
                metric="preferences_emo"
                )
    
    merged_hist(batches, title="Average preferences for lip sync: EMOTE vs ablation models",
                output_path=Path(__file__).parent / "study_result_1_lip.pdf",
                metric="preferences_lip"
                )


model_name_dict = {
    '2023_05_16_23-13-12_-2523817769843276359_FaceFormer_MEADP_Awav2vec2_Elinear_DBertPriorDecoder_Seml_NPE_Tff_predEJ_LVmmmLmm' : 'EMOTE \wo disentanglement',
    '2023_05_18_01-27-11_7629119778539369902_FaceFormer_MEADP_Awav2vec2_Elinear_DBertPriorDecoder_Seml_NPE_Tff_predEJ_LVmmmLmm' : 'EMOTE \wo video emotion loss',
    '2023_05_18_01-28-06_-6355446600867862848_FaceFormer_MEADP_Awav2vec2_Elinear_DBertPriorDecoder_Seml_NPE_Tff_predEJ_LVmmmLmm' : 'EMOTE \wo lip reading loss',
    '2023_05_16_20-26-30_5452619485602726463_FaceFormer_MEADP_Awav2vec2_Elinear_DFlameBertDecoder_Seml_PPE_Tff_predEJ_LVmmmLmm' : 'EMOTE \wo FLINT',
    '2023_05_13_21-00-49_-6819445356403438364_FaceFormer_MEADP_Awav2vec2T_Elinear_DBertPriorDecoder_Seml_NPE_predEJ_LVm' : 'EMOTE \wo any perceptual losses',
    '2023_05_18_01-58-31_6242149878645900496_FaceFormer_MEADP_Awav2vec2_Elinear_DBertPriorDecoder_Seml_NPE_Tff_predEJ_LVmEmmLmm' : 'EMOTE \w static emotion loss',
    '2023_05_10_13-10-08_8067654090108546902_FaceFormer_MEADP_Awav2vec2T_Elinear_DFaceFormerDecoder_Seml_PPE_predV_LV' : 'FaceFormer-EMO',
}
    

def merged_hist(batches, title, metric, order=None, with_std=False, fig=None, output_path=None):
    # Suppose we have n models and their results
    n = len(batches)  # replace with your number of models

    # Define the number of bins and the bar width
    bins = np.linspace(-4, 4, 5)  # change bin numbers as needed
    width = (bins[1] - bins[0]) / (n + 1)  # adjust for the number of models
    order = order or list(range(n))

    sns.set_style("whitegrid")
    colors = sns.color_palette("Set2", n)

    if fig is None:
        fig = plt.figure(figsize=(8,6))

    # Plot the results of each model
    # for i, batch in enumerate(batches):
    for i, order_idx in enumerate(order):
        result = batches[order_idx][f"avg_{metric}"]
        x_coords = bins - width*(n/2) + i*width
        plt.bar(x_coords, result, width=width, 
                label=f'Ours vs  {model_name_dict[ batches[order_idx]["model_b"]]}', color=colors[order_idx])
        if with_std:
            std = batches[order_idx][f"std_{metric}"]
            plt.errorbar(x_coords, result, yerr=std, fmt='none', color='k')

    # Add legend and labels
    plt.legend()
    # plt.xlabel('Value')
    plt.ylabel('Participant average preference')
    plt.title(title)
    fig.gca().set_xticklabels(["", "Strongly\n ours", "Weakly\n ours", "Indifferent", "Weakly\n other", "Stongly\n other"])
    # disable x ticks but keep the labels 
    plt.tick_params(axis='x', which='both', length=0)
    plt.show()
    
    if output_path is not None:
        # export to pdf, no white borders
        fig.savefig(output_path, bbox_inches='tight', pad_inches=0)
    


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
