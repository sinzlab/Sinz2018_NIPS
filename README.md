# Stimulus domain transfer in recurrent models for large scale cortical population prediction on video (Code)
Code to reproduce results of the NIPS 2018 paper: "Stimulus domain transfer in recurrent models for large scale cortical population prediction on video". 

## Requirements

* `docker`, `nvidia-docker` (version 1), and `nvidia-docker-compose`. You can easily run it with `nvidia-docker` version 2. In that case have a look at the `nvidia-docker-compose.yml.jinja` and extract the options for the `notebook` service to start the container. 
* [GIN](https://web.gin.g-node.org/G-Node/Info/wiki/GinCli#quickstart) along with `git` and `git-annex` to download the data. 

## Quickstart

Go to a folder of you choice and type the following commands in a [shell of your choice](https://fishshell.com/):

```bash
git clone https://github.com/sinzlab/Sinz2018_NIPS.git
cd Sinz2018_NIPS

# get the data
gin get cajal/Sinz2018_NIPS_data # might take a while; fast internet recommended

# create a file with DB credentials
echo "DJ_HOST=archive.datajoint.io" >> .env
echo "DJ_USER=public" >> .env
echo "DJ_PASS=public-user" >> .env

# create docker container (possibly you need sudo)

```

