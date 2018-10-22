# Stimulus domain transfer in recurrent models for large scale cortical population prediction on video (Code)
Code to reproduce results of the NIPS 2018 paper: "Stimulus domain transfer in recurrent models for large scale cortical population prediction on video". 

## Requirements

* `docker`, `nvidia-docker` (version 1), and `nvidia-docker-compose`. You can easily run it with `nvidia-docker` version 2. In that case have a look at the `nvidia-docker-compose.yml.jinja` and extract the options for the `notebook` service to start the container. 
* [GIN](https://web.gin.g-node.org/G-Node/Info/wiki/GinCli#quickstart) along with `git` and `git-annex` to download the data. 

## Data License

The data shared with this code is  licensed under a This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/">Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License</a>. This license requires that you contact us before you use the data in your own research. In particular, this means that you have to ask for permission if you intend to publish a new analysis performed with this data (no derivative works-clause).

<a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-nd/4.0/88x31.png" /></a>

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
nvidia-docker-compose -t docker-compose.yml.jinja build notebook0
```

Then you can start the container via

```bash
nvidia-docker-compose -t docker-compose.yml.jinja up notebook0
```

Now you should be able to access the jupyter notebooks via `YOURCOMPUTER:2018` in the browser. 

## Custom Database Server

You can also run the notebook with your own database server. In that case you need to insert the content of `Sinz2018_NIPS_data/dbdump/nips2018.sql` into your own database and change the `DJ_HOST, DJ_USER, DJ_PASS` parameters accordingly. 

