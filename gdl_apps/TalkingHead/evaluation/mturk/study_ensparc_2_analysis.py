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
from gdl.datasets.AffectNetDataModule import AffectNetExpressions
import matplotlib.pyplot as plt
import matplotlib.cm as cm


path_to_result_csv = "/is/cluster/fast/scratch/rdanecek/studies/enspark_v2/results_ensparc_2_pilot.csv"
path_to_protocols = "/is/cluster/fast/scratch/rdanecek/studies/enspark_v2/study_2/"


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

    task_str = participant_results[-2].strip('"')
    task_list = task_str.split(";")
    # TODO: sanity check the task list with the protocol/csv file

    answers = participant_results[-1].strip().strip('"').split(";")
    # filter out empty strings 
    answers = [answer for answer in answers if answer != ""]
    answers_lip = answers

    num_repeats = protocol["num_repeats"]
    # flips = protocol["flips"][0]
    catch_trials = protocol["catch_trials"][0]
    # emotions = protocol["correct_emotions"][0]

    # if len(emotions) + num_repeats== len(catch_trials) :
    #     emotions = emotions + emotions[:num_repeats]
    # assert len(emotions) == len(catch_trials), "Number of emotions is different from the expected number of emotions."
    

    # assert len(answers_lip) == len(flips) == len(catch_trials), "Number of answers is different from the expected number of answers."
    
    if not len(answers_lip) == len(catch_trials): #== len(emotions):
        answers_lip = answers_lip[:len(catch_trials)]
        # # filter out by wrong number of answers
        # print("[WARNING]: Number of answers is different from the expected number of answers. Skipping participant")
        # return None, False

    model = protocol["model"]
    # model_b = protocol["model_b"]

    sanity_passed = False
    for task in task_list: 
        vid = task.strip('"')
        if model in vid :
            sanity_passed = True
            break
    assert sanity_passed, "The participant did not see the expected videos."
    
    if discard_repeats: 
        task_list = task_list[num_repeats:]
        answers_lip = answers_lip[num_repeats:]
        catch_trials = catch_trials[num_repeats:]
        # emotions = emotions[num_repeats:]

    conf_matrix_model = np.zeros((8,8))
    # conf_matrix_catch = np.zeros((8,8))

    correct_answers_per_emotion =  8*[0]
    incorrect_answers_per_emotion = 8*[0]
    emotion_gt_counter = 8*[0]
    emotion_answer_counter = 8*[0]
    num_emotions = 8

    caught_counter = 0

    for i in range(len(answers_lip)):
        answer = int(answers_lip[i])-1
        task = task_list[i]
        answered_emo = AffectNetExpressions(answer).name
        answered_idx = AffectNetExpressions(answer).value
        if catch_trials[i] == 1:
            correct_emo = Path(task).stem.split('_')[2]
            if correct_emo != answered_emo:
                caught_counter += 1
        else:
            # correct_emotion = emotions[i]
            # correct_emotion = correct_emotion[0].upper() + correct_emotion[1:]
            correct_emo = Path(task).parent.name.split("_")[1]
            # assert filename_emo == correct_emotion, "The emotion in the protocol does not match the emotion in the video filename."
            emo_idx  = AffectNetExpressions[correct_emo].value
            conf_matrix_model[emo_idx][answered_idx] += 1
            if correct_emo == answered_emo:
                correct_answers_per_emotion[emo_idx] += 1
            else:
                incorrect_answers_per_emotion[emo_idx] += 1
            emotion_gt_counter[emo_idx] += 1
            emotion_answer_counter[answered_idx] += 1
    caught_emo = caught_counter > 1
    conf_matrix_model = np.array(conf_matrix_model) / np.sum(conf_matrix_model, axis=1, keepdims=True)
    emotion_gt_counter = np.array(emotion_gt_counter)
    return  conf_matrix_model, \
        np.array(correct_answers_per_emotion), np.array(incorrect_answers_per_emotion), \
        emotion_gt_counter, np.array(emotion_answer_counter), \
        caught_emo

    
def analyze_single_batch(header, result_lines, protocol):
    # assignment_label = '"AssignmentId"'
    # assignment_idx = header.index(assignment_label)
    # find all the results with the same assignment id
    # assignment_results = [result for result in result_lines if result.split(",")[assignment_idx] == assignment_id]
    assignment_results = result_lines
    # analyze the participant
    participant_results = []
    summed_ratings_lip = np.array([0] * 5)
    summed_ratings_per_emotion = np.zeros((8, 5))
    ratings_lip = []
    ratings_per_emotion = []
    num_participants = len(assignment_results)
    num_useful_participants = len(assignment_results)
    conf_matrix_list = []
    conf_matrix_sum = np.zeros((8,8))
    correct_answers_per_emotion_list = []
    incorrect_answers_per_emotion_list = []
    correct_answers_per_emotion_sum = np.zeros((8))
    incorrect_answers_per_emotion_sum = np.zeros((8))

    for ri, result in enumerate(assignment_results):
        conf_matrix_participant, \
        correct_answers_per_emotion, incorrect_answers_per_emotion, \
        emotion_gt_counter, emotion_answer_counter, \
        caught_emo = analyze_participant(header, result, protocol)
        if caught_emo:
            num_useful_participants -= 1
            continue
        conf_matrix_list += [conf_matrix_participant]
        conf_matrix_sum += conf_matrix_participant
        correct_answers_per_emotion_list += [correct_answers_per_emotion]
        incorrect_answers_per_emotion_list += [incorrect_answers_per_emotion]
        correct_answers_per_emotion_sum += correct_answers_per_emotion
        incorrect_answers_per_emotion_sum += incorrect_answers_per_emotion
    total_accuracy_per_emotion = correct_answers_per_emotion_sum / (correct_answers_per_emotion_sum + incorrect_answers_per_emotion_sum)
    total_accuracy = np.sum(correct_answers_per_emotion_sum) / np.sum(correct_answers_per_emotion_sum + incorrect_answers_per_emotion_sum)
    
    conf_matrix_sum = conf_matrix_sum / num_useful_participants

    return total_accuracy, total_accuracy_per_emotion, conf_matrix_sum, num_participants, num_useful_participants





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
    
    conf_mats = []
    for batch_i, (protocol_name, batch_protocol) in enumerate (protocol.items()):
        # protocol is an oredered dict, get the protocol by index 
        
        # num_participants = batch_protocol["num_participants"]

        batch_results = result_lines[batch_i * num_participants : (batch_i + 1) * num_participants]

        total_accuracy, total_accuracy_per_emotion, conf_matrix_sum, num_participants, num_useful_participants \
            = analyze_single_batch(header, batch_results, batch_protocol)
        batches[batch_i] = {
            'protocol_name' : protocol_name,
            'model' : batch_protocol["model"],
            'total_accuracy' : total_accuracy, 
            'total_accuracy_per_emotion' : total_accuracy_per_emotion, 
            'num_participants' : num_participants, 
            'num_useful_participants' : num_useful_participants
        }
        # print resutls for this batch
        print("Batch: ", batch_i)
        print("Model: ", batch_protocol["model"])
        print("Number of participants: ", num_participants)
        print("Number of useful participants: ", num_useful_participants)
        print("Total accuracy: ", total_accuracy)
        print("Total accuracy per emotion: ", total_accuracy_per_emotion)
        for i in range(8):
            print("Accuracy for emotion ", AffectNetExpressions(i).name, ": ", total_accuracy_per_emotion[i])
        # pretty print the confusion matrix
        print("Confusion matrix: ")
        print(conf_matrix_sum)

        conf_mats += [conf_matrix_sum]

    # stack the conf mats
    conf_matrix_model = np.stack(conf_mats, axis=-1)

    # plot the conf mats
    plot_confusion_matrix(conf_matrix_model, ["ours", "baseline"], class_names=[AffectNetExpressions(i).name for i in range(8)])
    print("Done")




def plot_confusion_matrix(data, model_names, class_names):
    # Create a figure to hold the subplots
    fig, axes = plt.subplots(8, 8, figsize=(20, 20))
    colors = cm.get_cmap('tab10', len(model_names)).colors
    # Iterate over the confusion matrix
    for i in range(8):
        for j in range(8):
            # Extract the data for this cell
            cell_data = data[i, j]
            
            # Create a bar plot for this cell
            axes[i, j].barh(model_names, cell_data, color=colors)
            
            # Optionally, add some labels or title
            # axes[i, j].set_title(f'Cell {i+1}, {j+1}')
            # axes[i, j].set_ylabel('Score')
            # axes[i, j].set_xlabel('Model')
            # Optionally, add some labels or title
            # axes[i, j].set_title(f'Predicted: {class_names[j]}; True: {class_names[i]}', fontsize=8)
             # Remove titles for each cell and set x and y labels only for outer cells
            if j == 0:
                axes[i, j].set_ylabel(f'T: {class_names[i]}', fontsize=12)
            if i == 0:
                axes[i, j].set_title(f'P: {class_names[j]}', fontsize=12)
    
 
            # axes[i, j].set_ylabel('Score')
            axes[i, j].set_yticks(range(len(model_names)))
            axes[i, j].set_yticklabels(model_names, rotation='horizontal')
            
            # set limits for the Y axis
            axes[i, j].set_xlim([0, 1])
 
    
    # Improve layout
    # fig.tight_layout()
    # Increase space between subplots
    fig.subplots_adjust(hspace=1., wspace=1.0)

    plt.show()





# def plot_confusion_matrix(method_list, confusion_matrix, num_classes, class_labels,
#                           abs_conf_matrix=None, abs_class_counts=None):
#     if isinstance(num_classes, int):
#         num_classes_1 = num_classes
#         num_classes_2 = num_classes
#     else:
#         num_classes_1 = num_classes[0]
#         num_classes_2 = num_classes[1]

#     # if not isinstance(class_labels, list):
#     class_labels_1 = class_labels
#     class_labels_2 = class_labels
#     # else:
#     #     class_labels_1 = class_labels[0]
#     #     class_labels_2 = class_labels[1]


#     fig, axes = plt.subplots(num_classes_1, num_classes_2, figsize=(20, 20))
#     colors = ['r', 'g', 'b', 'y', 'c', 'm', 'k']
#     # for each row and column
#     for i in range(num_classes_1):
#         # for each column
#         for j in range(num_classes_2):
#             # for each method, get the confusion matrix score
#             scores = []
#             names = []
#             for method in method_list:
#                 # get the score for the current method
#                 # score = total_rel_confusion_matrices[method][i, j]
#                 score = confusion_matrix[method][i, j]
#                 # append the score to the list
#                 scores.append(score)
#                 # append the method name to the list
#                 # if j != 0:
#                 #     names.append("")
#                 # else:
#                 names.append(method)

#             if i == 0:
#                 # set the subtitle to the emotion label
#                 if class_labels_1 is class_labels_2:
#                     axes[i, j].set_title(class_labels_2[j], fontsize=20)
#                 else:
#                     # [WARNING] UGLY LABEL HACK
#                     # axes[i, j].set_title(class_labels_2[j+2], fontsize=20)
#                     # axes[i, j].set_title(class_labels_1[j+1], fontsize=20)
#                     axes[i, j].set_title(class_labels_2[j], fontsize=20)

#             # create a bar plot, each bar has a unique color
#             # and the width is the score
#             # the y-axis is the method name
#             # the x-axis is the score
#             # create the bar plot
#             ax = axes[i, j]
#             barlist = ax.barh(names, scores)

#             # if j > 0, disable y-tick labels
#             if j > 0:
#                 ax.set_yticklabels([])
#             else:
#                 # set y label to the emotion label with a large font
#                 # ax.set_ylabel(class_labels_1[i+1], fontsize=20)
#                 if class_labels_1 is class_labels_2:
#                     ax.set_ylabel(class_labels_1[i], fontsize=20)
#                 else:
#                     # [WARNING] UGLY LABEL HACK
#                     ax.set_ylabel(class_labels_1[i], fontsize=20)
#                 # ax.set_ylabel(mapping[i+1])

#             # set x axis range to [0, 1]
#             ax.set_xlim([0, 1.25])

#             # set a unique color for each bar
#             for bi, bar in enumerate( barlist):
#                 bar.set_color(colors[bi % len(colors)])
#                 # add a text label to the right end of each bar
#                 # with the score as its value
#                 # ax.text(bar.get_width() + 0.01, bar.get_y() + 0.5,

#                 # height = bar.get_height()
#                 width = bar.get_width()
#                 # plt.text(bar.get_y() + bar.get_width() / 2.0, height, f'{height:.0f}', ha='center', va='bottom')
#                 # plt.text(bar.get_y() + bar.get_width() / 2.0, height, f'ha', ha='center', va='bottom')
#                 # plt.text(width, bar.get_x() + bar.get_height() / 2.0, f'ha', ha='center', va='bottom')
#                 # ax.text(bar.get_x() + bar.get_height() / 2.0, width, f'ha', ha='center', va='bottom')

#                 if abs_conf_matrix is not None:
#                     if not (class_labels_1 is class_labels_2):
#                         bar_label = f"{int(abs_conf_matrix[method_list[bi]][i, j])}/{int(abs_class_counts[method_list[bi]][i+1])}"
#                     else:
#                         bar_label = f"{int(abs_conf_matrix[method_list[bi]][i,j])}/{int(abs_class_counts[method_list[bi]][i])}"
#                     ax.text(bar.get_width() + 0.15, bar.get_y(), bar_label, ha='center', va='bottom')
#                 # ax.text(bar.get_width() + 0.05, bar.get_y(), f'ha', ha='center', va='bottom')

#             if class_labels_1 is class_labels_2:
#                 if i == j:
#                     #make the backround beige
#                     ax.set_facecolor('#f5f5dc')
#             else:
#                 if i == j-1:
#                     #make the backround beige
#                     ax.set_facecolor('#f5f5dc')

#     # set the title of the figure
#     # fig.suptitle("Relative Confusion Matrix Scores")
#     return fig


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
