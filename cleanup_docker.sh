#!/bin/bash

# Stop all running containers
if [ "$(sudo docker ps -q)" ]; then
  sudo docker stop $(sudo docker ps -q)
else
  echo "No running containers to stop."
fi

# Remove all containers
if [ "$(sudo docker ps -aq)" ]; then
  sudo docker rm $(sudo docker ps -aq)
else
  echo "No containers to remove."
fi

# Remove all images
if [ "$(sudo docker images -q)" ]; then
  sudo docker rmi -f $(sudo docker images -q)
else
  echo "No images to remove."
fi

echo "Docker cleanup completed."
