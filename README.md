# mediapipe-stuff
This is an attempt to build a body and face tracking application using python and media-pipe.

This application is ideal for use in a docker container that can be deployed on a GPU rental site like RunPod or Vast.

A Docker image is available at https://hub.docker.com/repository/docker/khou163/mediapipe-vast with all the prerequisites needed to run this.  This image only contains the mediapipe script to track body position, not facial features.  Regretably, the original dockerfile was lost due to a computer issue, but I will be working to reconstruct it.
