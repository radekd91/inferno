"""
Author: Radek Danecek
Copyright (c) 2022, Radek Danecek
All rights reserved.

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms 
# in the LICENSE file included with this software distribution. 
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2022 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at emoca@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de
"""


from inferno.datasets.FaceVideoDataModule import FaceVideoDataModule
from inferno.datasets.AffectNetDataModule import AffectNetExpressions
from inferno.datasets.EmotionalImageDataset import EmotionalImageDataset
from enum import Enum
import pickle as pkl
from pathlib import Path
import hashlib
from tqdm import auto
import numpy as np
import PIL
from collections import OrderedDict


class Expression7(Enum):
    Neutral = 0
    Anger = 1
    Disgust = 2
    Fear = 3
    Happiness = 4
    Sadness = 5
    Surprise = 6
    None_ = 7


def affect_net_to_expr7(aff : AffectNetExpressions) -> Expression7:
    # try:
    if aff == AffectNetExpressions.Happy:
        return Expression7.Happiness
    if aff == AffectNetExpressions.Sad:
        return Expression7.Sadness
    if aff == AffectNetExpressions.Contempt:
        return Expression7.None_
    return Expression7[aff.name]
    # except KeyError as e:
    #     return Expression7.None_


def expr7_to_affect_net(expr : Expression7) -> AffectNetExpressions:
    # try:
    if isinstance(expr, int) or isinstance(expr, np.int32) or isinstance(expr, np.int64):
        expr = Expression7(expr)
    if expr == Expression7.Happiness:
        return AffectNetExpressions.Happy
    if expr == Expression7.Sadness:
        return AffectNetExpressions.Sad
    return AffectNetExpressions[expr.name]
    # except KeyError as e:
    #     return AffectNetExpressions.None_


class AU8(Enum):
    AU1 = 0
    AU2 = 1
    AU4 = 2
    AU6 = 3
    AU12 = 4
    AU15 = 5
    AU20 = 6
    AU25 = 7


class AffWild2DMBase(FaceVideoDataModule):
    """
    A data module which implements a wrapper for the AffWild2 dataset.
    https://ibug.doc.ic.ac.uk/resources/aff-wild2/ 
    """

    def _get_processed_annotations_for_sequence(self, sid):
        pass
        video_file = self.video_list[sid]
        suffix = Path(video_file.parts[-4]) / 'detections' / video_file.parts[-2]
        annotation = Path(self.root_dir / suffix) / "valid_annotations.pkl"
        emotions, valence, arousal, detections_fnames = FaceVideoDataModule._load_face_emotions(annotation)
        return emotions, valence, arousal, detections_fnames

    def _get_max_faces_per_image(self): 
        return 10

    def _create_emotional_image_dataset(self,
                                        annotation_list=None,
                                        filter_pattern=None,
                                        with_landmarks=False,
                                        with_segmentation=False,
                                        crash_on_missing_file=False):
        annotation_list = annotation_list or ['va', 'expr7', 'au8']
        detections_all = []
        annotations_all = OrderedDict()
        for a in annotation_list:
            annotations_all[a] = []
        recognition_labels_all = []


        import re
        if filter_pattern is not None:
            # p = re.compile(filter_pattern)
            p = re.compile(filter_pattern, re.IGNORECASE)

        for si in auto.tqdm(range(self.num_sequences)):
            sequence_name = self.video_list[si]

            if filter_pattern is not None:
                res = p.match(str(sequence_name))
                if res is None:
                    continue

            ## TODO: or more like an idea - a solution towards duplicate videos between va/au/expression set
            # would be to append the video path here to serve as a key in the dictionaries (instead of just the stem
            # of the path)

            detection_fnames, annotations, recognition_labels, discarded_annotations, detection_not_found = \
                self._get_validated_annotations_for_sequence(si, crash_on_failure=False)

            if detection_fnames is None:
                continue

            current_list = annotation_list.copy()
            for annotation_name, detection_list in detection_fnames.items():
                detections_all += detection_list
                # annotations_all += [annotations[key]]
                for annotation_key in annotations[annotation_name].keys():
                    if annotation_key in current_list:
                        current_list.remove(annotation_key)
                    array = annotations[annotation_name][annotation_key]
                    annotations_all[annotation_key] += array.tolist()
                    n = array.shape[0]

                recognition_labels_all += len(detection_list)*[annotation_name + "_" + str(recognition_labels[annotation_name])]
                if len(current_list) != len(annotation_list):
                    print("No desired GT is found. Skipping sequence %d" % si)

                for annotation_name in current_list:
                    annotations_all[annotation_name] += [None] * n

        print("Data gathered")
        print(f"Found {len(detections_all)} detections with annotations "
              f"of {len(set(recognition_labels_all))} identities")

        # #TODO: delete debug code:
        # N = 3000
        # detections = detections[:N] + detections[-N:]
        # recognition_labels_all = recognition_labels_all[:N] + recognition_labels_all[-N:]
        # for key in annotations_all.keys():
        #     annotations_all[key] = annotations_all[key][:N] + annotations_all[key][-N:]
        # # end debug code : todo remove

        invalid_indices = set()
        if not with_landmarks:
            landmarks = None
        else:
            landmarks = []
            print("Checking if every frame has a corresponding landmark file")
            for det_i, det in enumerate(auto.tqdm(detections_all)):
                lmk = det.parents[3]
                lmk = lmk / "landmarks" / (det.relative_to(lmk / "detections"))
                lmk = lmk.parent / (lmk.stem + ".pkl")
                file_exists = (self.output_dir / lmk).is_file()
                if not file_exists and crash_on_missing_file:
                    raise RuntimeError(f"Landmark does not exist {lmk}")
                elif not file_exists:
                    invalid_indices.add(det_i)
                landmarks += [lmk]

        if not with_segmentation:
            segmentations = None
        else:
            segmentations = []
            print("Checking if every frame has a corresponding segmentation file")
            for det_i, det in enumerate(auto.tqdm(detections_all)):
                seg = det.parents[3]
                seg = seg / "segmentations" / (det.relative_to(seg / "detections"))
                seg = seg.parent / (seg.stem + ".pkl")
                file_exists = (self.output_dir / seg).is_file()
                if not file_exists and crash_on_missing_file:
                    raise RuntimeError(f"Landmark does not exist {seg}")
                elif not file_exists:
                    invalid_indices.add(det_i)
                segmentations += [seg]

        invalid_indices = sorted(list(invalid_indices), reverse=True)
        for idx in invalid_indices:
            del detections_all[idx]
            del landmarks[idx]
            del segmentations[idx]
            del recognition_labels_all[idx]
            for key in annotations_all.keys():
                del annotations_all[key][idx]

        return detections_all, landmarks, segmentations, annotations_all, recognition_labels_all

    def get_annotated_emotion_dataset(self,
                                      annotation_list = None,
                                      filter_pattern=None,
                                      image_transforms=None,
                                      split_ratio=None,
                                      split_style=None,
                                      with_landmarks=False,
                                      with_segmentations=False,
                                      K=None,
                                      K_policy=None,
                                      # if you add more parameters here, add them also to the hash list
                                      load_from_cache=True # do not add this one to the hash list
                                      ):
        # Process the dataset
        str_to_hash = pkl.dumps(tuple([annotation_list, filter_pattern]))
        inter_cache_hash = hashlib.md5(str_to_hash).hexdigest()
        inter_cache_folder = Path(self.output_dir) / "cache" / str(inter_cache_hash)
        if (inter_cache_folder / "lists.pkl").exists() and load_from_cache:
            print(f"Found processed filelists in '{str(inter_cache_folder)}'. "
                  f"Reprocessing will not be needed. Loading ...")
            with open(inter_cache_folder / "lists.pkl", "rb") as f:
                detections = pkl.load(f)
                landmarks = pkl.load(f)
                segmentations = pkl.load(f)
                annotations = pkl.load(f)
                recognition_labels = pkl.load(f)
            print("Loading done")

        else:
            detections, landmarks, segmentations, annotations, recognition_labels = \
                self._create_emotional_image_dataset(
                    annotation_list, filter_pattern, with_landmarks, with_segmentations)
            inter_cache_folder.mkdir(exist_ok=True, parents=True)
            print(f"Dataset processed. Saving into: '{str(inter_cache_folder)}'.")
            with open(inter_cache_folder / "lists.pkl", "wb") as f:
                pkl.dump(detections, f)
                pkl.dump(landmarks, f)
                pkl.dump(segmentations, f)
                pkl.dump(annotations, f)
                pkl.dump(recognition_labels, f)
            print(f"Saving done.")

        if split_ratio is not None and split_style is not None:

            hash_list = tuple([annotation_list,
                               filter_pattern,
                               split_ratio,
                               split_style,
                               with_landmarks,
                               with_segmentations,
                               K,
                               K_policy,
                               # add new parameters here
                               ])
            cache_hash = hashlib.md5(pkl.dumps(hash_list)).hexdigest()
            cache_folder = Path(self.output_dir) / "cache" / "tmp" / str(cache_hash)
            cache_folder.mkdir(exist_ok=True, parents=True)
            # load from cache if exists
            if load_from_cache and (cache_folder / "lists_train.pkl").is_file() and \
                (cache_folder / "lists_val.pkl").is_file():
                print(f"Dataset split found in: '{str(cache_folder)}'. Loading ...")
                with open(cache_folder / "lists_train.pkl", "rb") as f:
                    # training
                     detection_train = pkl.load(f)
                     landmarks_train = pkl.load(f)
                     segmentations_train = pkl.load(f)
                     annotations_train = pkl.load(f)
                     recognition_labels_train = pkl.load(f)
                     idx_train = pkl.load(f)
                with open(cache_folder / "lists_val.pkl", "rb") as f:
                    # validation
                     detection_val = pkl.load(f)
                     landmarks_val = pkl.load(f)
                     segmentations_val = pkl.load(f)
                     annotations_val = pkl.load(f)
                     recognition_labels_val = pkl.load(f)
                     idx_val = pkl.load(f)
                print("Loading done")
            else:
                print(f"Splitting the dataset. Split style '{split_style}', split ratio: '{split_ratio}'")
                if image_transforms is not None:
                    if not isinstance(image_transforms, list) or len(image_transforms) != 2:
                        raise ValueError("You have to provide image transforms for both trainng and validation sets")
                idxs = np.arange(len(detections), dtype=np.int32)
                if split_style == 'random':
                    np.random.seed(0)
                    np.random.shuffle(idxs)
                    split_idx = int(idxs.size * split_ratio)
                    idx_train = idxs[:split_idx]
                    idx_val = idxs[split_idx:]
                elif split_style == 'manual':
                    idx_train = []
                    idx_val = []
                    for i, det in enumerate(auto.tqdm(detections)):
                        if 'Train_Set' in str(det):
                            idx_train += [i]
                        elif 'Validation_Set' in str(det):
                            idx_val += [i]
                        else:
                            idx_val += [i]

                elif split_style == 'sequential':
                    split_idx = int(idxs.size * split_ratio)
                    idx_train = idxs[:split_idx]
                    idx_val = idxs[split_idx:]
                elif split_style == 'random_by_label':
                    idx_train = []
                    idx_val = []
                    unique_labels = sorted(list(set(recognition_labels)))
                    np.random.seed(0)
                    print(f"Going through {len(unique_labels)} unique labels and splitting its samples into "
                          f"training/validations set randomly.")
                    for li, label in enumerate(auto.tqdm(unique_labels)):
                        label_indices = np.array([i for i in range(len(recognition_labels)) if recognition_labels[i] == label],
                                                 dtype=np.int32)
                        np.random.shuffle(label_indices)
                        split_idx = int(len(label_indices) * split_ratio)
                        i_train = label_indices[:split_idx]
                        i_val = label_indices[split_idx:]
                        idx_train += i_train.tolist()
                        idx_val += i_val.tolist()
                    idx_train = np.array(idx_train, dtype= np.int32)
                    idx_val = np.array(idx_val, dtype= np.int32)
                elif split_style == 'sequential_by_label':
                    idx_train = []
                    idx_val = []
                    unique_labels = sorted(list(set(recognition_labels)))
                    print(f"Going through {len(unique_labels)} unique labels and splitting its samples into "
                          f"training/validations set sequentially.")
                    for li, label in enumerate(auto.tqdm(unique_labels)):
                        label_indices = [i for i in range(len(recognition_labels)) if recognition_labels[i] == label]
                        split_idx = int(len(label_indices) * split_ratio)
                        i_train = label_indices[:split_idx]
                        i_val = label_indices[split_idx:]
                        idx_train += i_train
                        idx_val += i_val
                    idx_train = np.array(idx_train, dtype= np.int32)
                    idx_val = np.array(idx_val, dtype= np.int32)
                else:
                    raise ValueError(f"Invalid split style {split_style}")

                if split_ratio < 0 or split_ratio > 1:
                    raise ValueError(f"Invalid split ratio {split_ratio}")

                def index_list_by_list(l, idxs):
                    return [l[i] for i in idxs]

                def index_dict_by_list(d, idxs):
                    res = d.__class__()
                    for key in d.keys():
                        res[key] = [d[key][i] for i in idxs]
                    return res

                detection_train = index_list_by_list(detections, idx_train)
                annotations_train = index_dict_by_list(annotations, idx_train)
                recognition_labels_train = index_list_by_list(recognition_labels, idx_train)
                if with_landmarks:
                    landmarks_train = index_list_by_list(landmarks, idx_train)
                else:
                    landmarks_train = None

                if with_segmentations:
                    segmentations_train = index_list_by_list(segmentations, idx_train)
                else:
                    segmentations_train = None

                detection_val = index_list_by_list(detections, idx_val)
                annotations_val = index_dict_by_list(annotations, idx_val)
                recognition_labels_val = index_list_by_list(recognition_labels, idx_val)

                if with_landmarks:
                    landmarks_val = index_list_by_list(landmarks, idx_val)
                else:
                    landmarks_val = None

                if with_segmentations:
                    segmentations_val = index_list_by_list(segmentations, idx_val)
                else:
                    segmentations_val = None

                print(f"Dataset split processed. Saving into: '{str(cache_folder)}'.")
                with open(cache_folder / "lists_train.pkl", "wb") as f:
                    # training
                    pkl.dump(detection_train, f)
                    pkl.dump(landmarks_train, f)
                    pkl.dump(segmentations_train, f)
                    pkl.dump(annotations_train, f)
                    pkl.dump(recognition_labels_train, f)
                    pkl.dump(idx_train, f)
                with open(cache_folder / "lists_val.pkl", "wb") as f:
                    # validation
                    pkl.dump(detection_val, f)
                    pkl.dump(landmarks_val, f)
                    pkl.dump(segmentations_val, f)
                    pkl.dump(annotations_val, f)
                    pkl.dump(recognition_labels_val, f)
                    pkl.dump(idx_val, f)
                print(f"Saving done.")

            dataset_train = EmotionalImageDataset(
                detection_train,
                annotations_train,
                recognition_labels_train,
                image_transforms[0],
                self.output_dir,
                landmark_list=landmarks_train,
                segmentation_list=segmentations_train,
                K=K,
                K_policy=K_policy)

            dataset_val = EmotionalImageDataset(
                detection_val,
                annotations_val,
                recognition_labels_val,
                image_transforms[1],
                self.output_dir,
                landmark_list=landmarks_val,
                segmentation_list=segmentations_val,
                # K=K,
                K=1,
                # K=None,
                # K_policy=K_policy)
                K_policy='sequential')
                # K_policy=None)

            return dataset_train, dataset_val, idx_train, idx_val

        # dataset = EmotionalImageDataset(
        dataset = EmotionalImageDataset(
            detections,
            annotations,
            recognition_labels,
            image_transforms,
            self.output_dir,
            landmark_list=landmarks,
            segmentation_list=segmentations,
            K=K,
            K_policy=K_policy)
        return dataset

    def _draw_annotation(self, frame_draw : PIL.ImageDraw.Draw, val_gt : dict, font, color):
        all_str = ''
        for gt_type, val in val_gt.items():
            if gt_type == 'va':
                va_str = "V: %.02f  A: %.02f" % (val[0], val[1])
                all_str += "\n" + va_str
                # frame_draw.text((bb[1, 0] - 60, bb[0, 1] - 30,), va_str, font=fnt, fill=color)
            elif gt_type == 'expr7':
                frame_draw.text((bb[0, 0], bb[0, 1] - 30,), Expression7(val).name, font=font, fill=color)
            elif gt_type == 'au8':
                au_str = ''
                for li, label in enumerate(val):
                    if label:
                        au_str += AU8(li).name + ' '
                all_str += "\n" + au_str
                # frame_draw.text((bb[0, 0], bb[1, 1] + 30,), au_str, font=fnt, fill=color)
            else:
                raise ValueError(f"Unable to visualize this gt_type: '{gt_type}")
            frame_draw.text((bb[0, 0], bb[1, 1] + 10,), str(all_str), font=font, fill=color)


    def test_annotations(self, net=None, annotation_list = None, filter_pattern=None):
        net = net or self._get_emonet(self.device)

        dataset = self.get_annotated_emotion_dataset(annotation_list, filter_pattern)


    def _assign_gt_to_detections(self):
        for sid in range(self.num_sequences):
            self.assign_gt_to_detections_sequence(sid)



    def assign_gt_to_detections_sequence(self, sequence_id):
        print(f"Assigning GT to sequence {sequence_id}")

        def second_most_frequent_label():
            if len(most_frequent_labels) == 2:
                second_label = most_frequent_labels[1]
            elif len(most_frequent_labels) > 2:
                raise RuntimeError(f"Too many labels occurred with the same frequency. Unclear which one to pick.")
            else:
                most_frequent_count2 = list(counts2labels.keys())[1]
                most_frequent_labels2 = counts2labels[most_frequent_count2]
                if len(most_frequent_labels2) != 1:
                    raise RuntimeError(
                        f"Too many labels occurred with the same frequency. Unclear which one to pick.")
                second_label = most_frequent_labels2[0]
            return second_label

        def correct_left_right_order(left_center, right_center):
            left_right_dim = 0 # TODO: verify if this is correct
            if left_center[left_right_dim] < right_center[left_right_dim]:
                # left is on the left
                return 1
            elif left_center[left_right_dim] == right_center[left_right_dim]:
                # same place
                return 0
            # left is on the right
            return -1

        # detection_fnames = self._get_path_to_sequence_detections(sequence_id)
        # full_frames = self._get_frames_for_sequence(sequence_id)
        annotations = self._get_annotations_for_sequence(sequence_id)
        if len(annotations) == 0:
            print(f"No GT available for video '{self.video_list[sequence_id]}'")
            return
        annotation_type = annotations[0].parent.parent.parent.stem
        if annotation_type == 'AU_Set':
            anno_type = 'au8' # AU1,AU2,AU4,AU6,AU12,AU15,AU20,AU25
        elif annotation_type == 'Expression_Set':
            anno_type = 'expr7' # Neutral,Anger,Disgust,Fear,Happiness,Sadness,Surprise
        elif annotation_type == 'VA_Set':
            anno_type = 'va' # valence arousal -1 to 1
        else:
            raise ValueError(f"Unsupported annotation type: '{annotation_type}'")

        # load the recognitions:
        # recognition_file = self._get_recognition_filename(
        #     sequence_id, self.get_default_recognition_threshold())
        # indices, labels, mean, cov, recognition_fnames = FaceVideoDataModule._load_recognitions(
        #     recognition_file)
        indices, labels, mean, cov, recognition_fnames = self._get_recognition_for_sequence(sequence_id)
        counts2labels = OrderedDict()
        for key, val in labels.items():
            if key == -1: # skip invalid outliers
                continue
            count = len(val)
            if count not in counts2labels.keys():
                counts2labels[count] = []
            counts2labels[count] += [key]

        recognition_label_dict = OrderedDict()
        annotated_detection_fnames = OrderedDict()
        validated_annotations = OrderedDict()
        discarded_annotations = OrderedDict()
        detection_not_found = OrderedDict()

        # suffs = [str(Path(str(anno)[len(str(anno.parent / self.video_list[sequence_id].stem)):]).stem) for anno in
        #      annotations]
        suffs = [str(anno.stem)[len(str(self.video_list[sequence_id].stem)):] for anno in
             annotations]

        ### WARNING: HORRIBLE THINGS FOLLOW, PUT ON YOUR PROTECTIVE GOGGLES BEFORE YOU PROCEED
        # this next section is a ugly rule-based approach to assign annotation files to detected and recognized
        # faces. This assignment is not provided by the authors of aff-wild2 and therefore it's approximated
        # using these rules that are taken from the readme.

        # THERE IS ONLY ONE DOMINANT DETECTION AND ONE ANNOTATION FILE:
        if len(annotations) == 1 and suffs[0] == '':
            most_frequent_count = list(counts2labels.keys())[0]
            most_frequent_labels = counts2labels[most_frequent_count]

            if len(most_frequent_labels) != 1:
                raise ValueError("There seem to be two people at the same time in all pictures but we only "
                                 "have annotation for one")

            main_label = most_frequent_labels[0]
            main_detection_file_names = recognition_fnames[main_label]
            main_annotation_file = annotations[0]
            main_valid_detection_list, main_valid_annotation_list, main_discarded_list, main_detection_not_found_list \
                = self._map_detections_to_gt(main_detection_file_names, main_annotation_file, anno_type)

            recognition_label_dict[main_annotation_file.stem] = main_label
            annotated_detection_fnames[main_annotation_file.stem] = main_valid_detection_list
            validated_annotations[main_annotation_file.stem] = main_valid_annotation_list
            discarded_annotations[main_annotation_file.stem] = main_discarded_list
            detection_not_found[main_annotation_file.stem] = main_detection_not_found_list


            # THERE ARE TWO DOMINANT DETECTIONS BUT ONLY ONE IS ANNOTATED
        elif len(annotations) == 1 and (suffs[0] == '_left' or suffs[0] == '_right'):

            most_frequent_count = list(counts2labels.keys())[0]
            most_frequent_labels = counts2labels[most_frequent_count]

            detection_fnames, detection_centers, detection_sizes, _ = \
                self._get_detection_for_sequence(sequence_id)

            if len(most_frequent_labels) != 1:
                raise ValueError("There seem to be two people at the same time in all pictures but we only "
                                 "have annotation for one")

            main_label = most_frequent_labels[0]
            main_detection_file_names = recognition_fnames[main_label]
            main_annotation_file = annotations[0]
            main_valid_detection_list, main_valid_annotation_list, main_discarded_list, main_detection_not_found_list  \
                = self._map_detections_to_gt(main_detection_file_names, main_annotation_file, anno_type)

            other_label = second_most_frequent_label()
            other_detection_file_names = recognition_fnames[other_label]
            other_annotation_file = annotations[0] # use the same annotation, which one will be used is figured out next
            other_valid_detection_list, other_valid_annotation_list, other_discarded_list, other_detection_not_found_list\
                = self._map_detections_to_gt(other_detection_file_names, other_annotation_file, anno_type)

            other_center = self._get_bb_center_from_fname(other_detection_file_names[0], detection_fnames,
                                                          detection_centers)
            main_center = self._get_bb_center_from_fname(main_detection_file_names[0], detection_fnames,
                                                         detection_centers)
            if correct_left_right_order(other_center, main_center) == 1:
                pass # do nothing, order correct
            elif correct_left_right_order(other_center, main_center) == -1:
                # swap main and other
                print("Swapping left and right")
                other_label, main_label = main_label, other_label
                # other_valid_detection_list, main_valid_detection_list = main_valid_detection_list, other_valid_detection_list
                # other_valid_annotation_list, main_valid_annotation_list = main_valid_annotation_list, other_valid_annotation_list
            else:
                raise ValueError("Detections are in the same place. No way to tell left from right")

            # now other is on the left, and main is on the right, decide which one is annotated based on the suffix
            if suffs[0] == '_left':
                print("Choosing left")
                recognition_label_dict[other_annotation_file.stem] = other_label
                annotated_detection_fnames[other_annotation_file.stem] = other_valid_detection_list
                validated_annotations[other_annotation_file.stem] = other_valid_annotation_list
                discarded_annotations[other_annotation_file.stem] = other_discarded_list
                detection_not_found[other_annotation_file.stem] = other_detection_not_found_list
            else: # suffs[0] == '_right':
                print("Choosing right")
                recognition_label_dict[main_annotation_file.stem] = main_label
                annotated_detection_fnames[main_annotation_file.stem] = main_valid_detection_list
                validated_annotations[main_annotation_file.stem] = main_valid_annotation_list
                discarded_annotations[main_annotation_file.stem] = main_discarded_list
                detection_not_found[main_annotation_file.stem] = main_detection_not_found_list
        else:
            if len(suffs) > 2:
                print(f"Unexpected number of suffixes found {len(suffs)}")
                print(suffs)
                raise RuntimeError(f"Unexpected number of suffixes found {len(suffs)}")

            most_frequent_count = list(counts2labels.keys())[0]
            most_frequent_labels = counts2labels[most_frequent_count]

            detection_fnames, detection_centers, detection_sizes, _ = \
                self._get_detection_for_sequence(sequence_id)

            # THE CASE OF ONE DOMINANT DETECTION AND ONE SMALLER ONE (NO SUFFIX vs LEFT/RIGHT)
            if suffs[0] == '' and (suffs[1] == '_left' or suffs[1] == '_right'):
                if len(most_frequent_labels) != 1:
                    raise ValueError("There seem to be two people at the same time in all pictures but we only "
                                     "have annotation for one")

                main_label = most_frequent_labels[0]
                main_detection_file_names = recognition_fnames[main_label]
                main_annotation_file = annotations[0]
                main_valid_detection_list, main_valid_annotation_list, main_discarded_list, main_detection_not_found_list\
                    = self._map_detections_to_gt(main_detection_file_names, main_annotation_file, anno_type)

                recognition_label_dict[main_annotation_file.stem] = main_label
                annotated_detection_fnames[main_annotation_file.stem] = main_valid_detection_list
                validated_annotations[main_annotation_file.stem] = main_valid_annotation_list
                discarded_annotations[main_annotation_file.stem] = main_discarded_list
                detection_not_found[main_annotation_file.stem] = main_detection_not_found_list


                other_label = most_frequent_labels[1]
                other_detection_file_names = recognition_fnames[other_label]
                other_annotation_file = annotations[1]
                other_valid_detection_list, other_valid_annotation_list, other_discarded_list, other_detection_not_found_list \
                    = self._map_detections_to_gt(other_detection_file_names, other_annotation_file, anno_type)

                recognition_label_dict[other_annotation_file.stem] = other_label
                annotated_detection_fnames[other_annotation_file.stem] = other_valid_detection_list
                validated_annotations[other_annotation_file.stem] = other_valid_annotation_list
                discarded_annotations[other_annotation_file.stem] = other_discarded_list
                detection_not_found[other_annotation_file.stem] = other_detection_not_found_list

                other_center = self._get_bb_center_from_fname(other_detection_file_names[0], detection_fnames,
                                                        detection_centers)
                main_center = self._get_bb_center_from_fname(main_detection_file_names[0], detection_fnames,
                                                       detection_centers)
                if suffs[1] == '_left':
                    if correct_left_right_order(other_center, main_center) != 1:
                        raise RuntimeError("The main detection should be on the right and the other on the left but this is not the case")
                elif suffs[1] == '_right':
                    if correct_left_right_order(main_center, other_center) != 1:
                        raise RuntimeError(
                            "The main detection should be on the left and the other on the right but this is not the case")

            # THE CASE OF TWO ROUGHLY EQUALY DOMINANT DETECTIONS (LEFT and RIGHT)
            elif suffs[0] == '_left' and suffs[1] == '_right':
                #TODO: figure out which one is left and which one is right by loading the bboxes and comparing
                counts2labels.keys()
                left_label = most_frequent_labels[0]
                # if len(most_frequent_labels) == 2:
                #     right_label = most_frequent_labels[1]
                # elif len(most_frequent_labels) > 2:
                #     raise RuntimeError(f"Too many labels occurred with the same frequency. Unclear which one to pick.")
                # else:
                #     most_frequent_count2 = list(counts2labels.keys())[1]
                #     most_frequent_labels2 = counts2labels[most_frequent_count2]
                #     if len(most_frequent_labels2) != 1:
                #         raise RuntimeError(
                #             f"Too many labels occurred with the same frequency. Unclear which one to pick.")
                #     right_label = most_frequent_labels2[0]
                right_label = second_most_frequent_label()

                left_filename = recognition_fnames[left_label][0]
                left_center = self._get_bb_center_from_fname(left_filename, detection_fnames, detection_centers)

                right_filename = recognition_fnames[right_label][0]
                right_center = self._get_bb_center_from_fname(right_filename, detection_fnames, detection_centers)

                order = correct_left_right_order(left_center, right_center)
                # if left is not left, swap
                if order == -1:
                    left_label, right_label = right_label, left_label
                    left_filename, right_filename = right_filename, left_filename
                elif order == 0:
                    raise RuntimeError("Left and right detections have centers in the same place. "
                                       "No way to tell left from right")

                left_detection_file_names = recognition_fnames[left_label]
                left_annotation_file = annotations[0]
                left_valid_detection_list, left_annotation_list, left_discarded_list, left_detection_not_found_list \
                    = self._map_detections_to_gt(left_detection_file_names, left_annotation_file, anno_type)
                recognition_label_dict[left_annotation_file.stem] = left_label
                annotated_detection_fnames[left_annotation_file.stem] = left_valid_detection_list
                validated_annotations[left_annotation_file.stem] = left_annotation_list
                discarded_annotations[left_annotation_file.stem] = left_discarded_list
                detection_not_found[left_annotation_file.stem] = left_detection_not_found_list



                right_detection_file_names = recognition_fnames[right_label]
                right_annotation_file = annotations[1]

                right_valid_detection_list, right_valid_annotation_list, right_discarded_list, right_detection_not_found_list \
                    = self._map_detections_to_gt(right_detection_file_names, right_annotation_file, anno_type)
                recognition_label_dict[right_annotation_file.stem] = right_label
                annotated_detection_fnames[right_annotation_file.stem] = right_valid_detection_list
                validated_annotations[right_annotation_file.stem] = right_valid_annotation_list
                discarded_annotations[right_annotation_file.stem] = right_discarded_list
                detection_not_found[right_annotation_file.stem] = right_detection_not_found_list

                # THE FOLLOWING CASE SHOULD NEVER HAPPEN
            else:
                print(f"Unexpected annotation case found.")
                print(suffs)
                raise RuntimeError(f"Unexpected annotation case found: {str(suffs)}")

        out_folder = self._get_path_to_sequence_detections(sequence_id)
        out_file = out_folder / "valid_annotations.pkl"
        FaceVideoDataModule._save_annotations(out_file, annotated_detection_fnames, validated_annotations,
                                              recognition_label_dict, discarded_annotations, detection_not_found)


def main():
    # root = Path("/home/rdanecek/Workspace/mount/scratch/rdanecek/data/aff-wild2/")
    # root = Path("/is/cluster/work/rdanecek/data/aff-wild2/")
    root = Path("/ps/project/EmotionalFacialAnimation/data/aff-wild2/")
    root_path = root / "Aff-Wild2_ready"
    output_path = root / "processed"
    # subfolder = 'processed_2020_Dec_21_00-30-03'
    subfolder = 'processed_2021_Jan_19_20-25-10'
    dm = AffWild2DMBase(str(root_path), str(output_path), processed_subfolder=subfolder)
    dm.prepare_data()
    #
    # test_videos = [
    #     '9-15-1920x1080.mp4', # smiles, sadness, tears, girl with glasses
    #     '19-24-1920x1080.mp4', # angry young black guy on stage
    #     '17-24-1920x1080.mp4', # black guy on stage, difficult light
    #     '23-24-1920x1080.mp4', # white woman, over-articulated expressions
    #     '24-30-1920x1080-2.mp4', # white woman, over-articulated expressions
    #     '28-30-1280x720-1.mp4', # angry black guy
    #     '31-30-1920x1080.mp4', # crazy white guy, beard, view from the side
    #     '34-25-1920x1080.mp4', # white guy, mostly neutral
    #     '50-30-1920x1080.mp4', # baby
    #     '60-30-1920x1080.mp4', # smiling asian woman
    #     '61-24-1920x1080.mp4', # very lively white woman
    #     '63-30-1920x1080.mp4', # smiling asian woman
    #     '66-25-1080x1920.mp4', # white girl acting out an emotional performance
    #     '71-30-1920x1080.mp4', # old white woman, camera shaking
    #     '83-24-1920x1080.mp4', # excited black guy (but expressions mostly neutral)
    #     '87-25-1920x1080.mp4', # white guy explaining stuff, mostly neutral
    #     '95-24-1920x1080.mp4', # white guy explaining stuff, mostly neutral
    #     '122-60-1920x1080-1.mp4', # crazy white youtuber, lots of overexaggerated expressiosn
    #     '135-24-1920x1080.mp4', # a couple watching a video, smiles, sadness, tears
    #     '82-25-854x480.mp4', # Rachel McAdams, sadness, anger
    #     '111-25-1920x1080.mp4', # disgusted white guy
    #     '121-24-1920x1080.mp4', # white guy scared and happy faces
    # ]
    #
    # indices = [dm.video_list.index(Path('VA_Set/videos/Train_Set') / name) for name in test_videos]
    #
    # # i = dm.video_list.index(Path('VA_Set/videos/Train_Set/119-30-848x480.mp4')) # black lady with at Oscars
    # # i = dm.video_list.index(Path('VA_Set/videos/Train_Set/9-15-1920x1080.mp4')) # black lady with at Oscars
    # i = dm.video_list.index(Path('VA_Set/videos/Train_Set/119-30-848x480.mp4')) # black lady with at Oscars
    # i = dm.video_list.index(Path('VA_Set/videos/Train_Set/119-30-848x480.mp4')) # black lady with at Oscars
    # i = dm.video_list.index(Path('VA_Set/videos/Train_Set/119-30-848x480.mp4')) # black lady with at Oscars
    # i = dm.video_list.index(Path('VA_Set/videos/Train_Set/119-30-848x480.mp4')) # black lady with at Oscars
    # i = dm.video_list.index(Path('VA_Set/videos/Train_Set/119-30-848x480.mp4')) # black lady with at Oscars
    # i = dm.video_list.index(Path('VA_Set/videos/Train_Set/119-30-848x480.mp4')) # black lady with at Oscars
    # i = dm.video_list.index(Path('Expression_Set/videos/Train_Set/1-30-1280x720.mp4')) # black lady with at Oscars
    # # dm._process_everything_for_sequence(i)
    # # dm._detect_faces_in_sequence(i)
    # # dm._segment_faces_in_sequence(i)

    # dm._extract_emotion_from_faces_in_sequence(0)

    # rpoblematic indices
    # dm._segment_faces_in_sequence(30)
    # dm._segment_faces_in_sequence(156)
    # dm._segment_faces_in_sequence(399)

    # dm._create_emotional_image_dataset(['va'], "VA_Set")
    # dm._recognize_emotion_in_sequence(0)
    # i = dm.video_list.index(Path('AU_Set/videos/Train_Set/130-25-1280x720.mp4'))
    # i = dm.video_list.index(Path('AU_Set/videos/Train_Set/52-30-1280x720.mp4'))
    # i = dm.video_list.index(Path('Expression_Set/videos/Train_Set/46-30-484x360.mp4'))
    # i = dm.video_list.index(Path('Expression_Set/videos/Train_Set/135-24-1920x1080.mp4'))
    # i = dm.video_list.index(Path('VA_Set/videos/Train_Set/30-30-1920x1080.mp4'))
    # dm._recognize_faces_in_sequence(i)
    # dm._identify_recognitions_for_sequence(i)
    # for i in range(7,8):
    # for i in range(8, 30):
    #     dm._recognize_faces_in_sequence(i, num_workers=8)
    #     dm._identify_recognitions_for_sequence(i)
    #     print("----------------------------------")
    #     print(f"Assigning GT to detections for seq: {i}")
    #     dm.assign_gt_to_detections_sequence(i)
    # dm._detect_faces()
    # dm._detect_faces_in_sequence(30)
    # dm._detect_faces_in_sequence(107)
    # dm._detect_faces_in_sequence(399)
    # dm._detect_faces_in_sequence(21)
    # dm.create_reconstruction_video_with_recognition_and_annotations(100, overwrite=True)
    # dm._identify_recognitions_for_sequence(0)
    # dm.create_reconstruction_video_with_recognition(0, overwrite=True)
    # dm._identify_recognitions_for_sequence(0, distance_threshold=1.0)
    # dm.create_reconstruction_video_with_recognition(0, overwrite=True, distance_threshold=1.0)
    # dm._gather_detections()

    # failed_jobs = [48,  83, 102, 135, 152, 153, 154, 169, 390]
    # failed_jobs = [48,  83, 102] #, 135, 152, 153, 154, 169, 390]
    # failed_jobs = [135, 152, 153] #, 154, 169, 390]
    # failed_jobs = [154, 169, 390]
    # for fj in failed_jobs:

    fj = 9
    # dm._detect_faces_in_sequence(fj)
    # dm._recognize_faces_in_sequence(fj)
    retarget_from = None
    # retarget_from = "/ps/project/EmotionalFacialAnimation/data/aff-wild2/processed/" \
    #                 "processed_2021_Jan_19_20-25-10/AU_Set/detections/Test_Set/82-25-854x480/000001_000.png" ## Rachel McAdams
    # retarget_from = "/ps/project/EmotionalFacialAnimation/data/aff-wild2/processed/processed_2021_Jan_19_20-25-10/AU_Set/detections/Test_Set/30-30-1920x1080/000880_000.png" # benedict
    # retarget_from = "/ps/project/EmotionalFacialAnimation/data/aff-wild2/processed/processed_2021_Jan_19_20-25-10/AU_Set/detections/Train_Set/11-24-1920x1080/000485_000.png" # john cena
    # retarget_from = "/ps/project/EmotionalFacialAnimation/data/aff-wild2/processed/processed_2021_Jan_19_20-25-10/AU_Set/detections/Train_Set/26-60-1280x720/000200_000.png" # obama
    # retarget_from = "/ps/project/EmotionalFacialAnimation/data/random_images/soubhik.jpg" # obama
    # dm._reconstruct_faces_in_sequence(fj, rec_method="emoca", retarget_from=retarget_from, retarget_suffix="soubhik")
    dm._reconstruct_faces_in_sequence(fj, rec_method="emoca", retarget_from=retarget_from, retarget_suffix="_retarget_cena")
    # dm._reconstruct_faces_in_sequence(fj, rec_method='deep3dface')
    # dm.create_reconstruction_video(fj, overwrite=False)
    # dm.create_reconstruction_video(fj, overwrite=False, rec_method='emoca')
    dm.create_reconstruction_video(fj, overwrite=True, rec_method='emoca', retarget_suffix="_retarget_soubhik")
    dm.create_reconstruction_video(fj, overwrite=True, rec_method='emoca', retarget_suffix="_retarget_obama")
    dm.create_reconstruction_video(fj, overwrite=True, rec_method='emoca', retarget_suffix="_retarget_cumberbatch")
    dm.create_reconstruction_video(fj, overwrite=True, rec_method='emoca', retarget_suffix="_retarget_cena")
    # dm.create_reconstruction_video(fj, overwrite=False, rec_method='emoca', image_type="coarse")
    # dm.create_reconstruction_video(fj, overwrite=False, rec_method='deep3dface')
    # dm.create_reconstruction_video(fj, overwrite=False, rec_method='deep3dface')
    # dm.create_reconstruction_video_with_recognition(fj, overwrite=True)
    # dm._identify_recognitions_for_sequence(fj)
    # dm.create_reconstruction_video_with_recognition(fj, overwrite=True, distance_threshold=0.6)

    # dm._recognize_faces_in_sequence(400)
    # dm._reconstruct_faces_in_sequence(400)
    print("Peace out")


if __name__ == "__main__":
    main()