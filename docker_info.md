# Setting up Docker

### Docker installation on Linux

* Removing current packages:
  * `sudo apt-get remove docker docker-engine docker.io containerd runc`
* Updating
```
sudo apt-get update
 sudo apt-get install \
    ca-certificates \
    curl \
    gnupg \
    lsb-release
 ```

* Adding docker public key
```
sudo mkdir -p /etc/apt/keyrings
 curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
```

```
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
```


```
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-compose-plugin
```

Getting versions:  `apt-cache madison docker-ce`



Installing:
```sudo apt-get install docker-ce=5:20.10.19~3-0~ubuntu-focal docker-ce-cli=5:20.10.19~3-0~ubuntu-focal containerd.io docker-compose-plugin```

## NVIDIA Docker
```curl https://get.docker.com | sh \
  && sudo systemctl --now enable docker
```

Key:
```
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
      && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
      && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```

Installing:
```
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
sudo service docker restart
```

Testing it:
```
sudo docker run --rm --gpus all nvidia/cuda:11.0.3-base-ubuntu20.04 nvidia-smi
```


Building env

```docker build -t graphite .```


Run the following inside the code repo:
```
docker run --ipc=host --gpus all -v $PWD:/workspace -v /data/pcqm4mv2_datahub:/workspace/data -p <port-host>:<port-docker> --hostname localhost --rm -it graphite /bin/bash 
```

For example:
```
docker run --ipc=host --gpus all -v $PWD:/workspace -v /data/pcqm4mv2_datahub:/workspace/data -p 4044:8888 --hostname localhost --rm -it graphite /bin/bash 
```

```
sudo groupadd docker
sudo usermod -aG docker $USER
newgrp docker
```
