import sys, os 
import math
sys.path = [os.path.abspath("../../..")] + sys.path

from pathlib import Path

if len(sys.argv) > 1:
    sid = int(sys.argv[1])
else:
    sid = 0


from inferno.datasets.AffectNetDataModule import AffectNetDataModule, AffectNetEmoNetSplitModule

print("Detecting mediapipe landmarks in subset %d" % sid)

## OLD AFFECTNET VERSION
#dm = AffectNetDataModule(
#         "/ps/project/EmotionalFacialAnimation/data/affectnet/",
#         "/ps/scratch/rdanecek/data/affectnet/",
#         processed_subfolder="processed_2021_Apr_05_15-22-18",
#         mode="manual",
#         scale=1.25)


# NEW AFFECTNET VERSION, LARGER BB (DOESN'T CUT FOREHEADS ANYMore))
dm = AffectNetDataModule(
             # "/home/rdanecek/Workspace/mount/project/EmotionalFacialAnimation/data/affectnet/",
             "/ps/project/EmotionalFacialAnimation/data/affectnet/",
             # "/home/rdanecek/Workspace/mount/scratch/rdanecek/data/affectnet/",
             # "/home/rdanecek/Workspace/mount/work/rdanecek/data/affectnet/",
             "/is/cluster/work/rdanecek/data/affectnet/",
            #  processed_subfolder="processed_2021_Aug_27_19-58-02",
             processed_subfolder="processed_2021_Apr_05_15-22-18",
             processed_ext=".png",
             mode="manual",
             scale= 1.25,
             image_size=224,
            #  bb_center_shift_x=0,
            #  bb_center_shift_y=-0.3,
             ignore_invalid=True,
            )

num_subsets = math.ceil( len(dm.df)/ dm.subset_size)
print(f"Processing subset {sid}/{num_subsets}")

dm._detect_landmarks_mediapipe(dm.subset_size * sid, min((sid + 1) * dm.subset_size, len(dm.df)))

print("Finished decting faces")