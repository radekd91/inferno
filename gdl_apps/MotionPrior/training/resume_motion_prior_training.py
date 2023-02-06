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


from gdl_apps.MotionPrior.training.train_motion_prior import resume_training
import sys, os


def main():
    resume_from = sys.argv[1]
    stage = int(sys.argv[2])
    resume_from_previous = bool(int(sys.argv[3]))
    force_new_location = bool(int(sys.argv[4]))

    resume_training(resume_from, start_at_stage=stage,
                    resume_from_previous=resume_from_previous, force_new_location=force_new_location)


if __name__ == "__main__":
    main()

