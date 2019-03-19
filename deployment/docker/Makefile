IMAGE_VERSION=0.1123

# Makefile - debugger command
#   $ cat -e -t -v  makefile_name


help:	## Help Command
	@fgrep -h "##" $(MAKEFILE_LIST) | fgrep -v fgrep | sed -e 's/\\$$//' | sed -e 's/##//'

version: ## Prints Build version
	set -e; \
	export IMAGE_VERSION=${IMAGE_VERSION}
	@echo "Image Version "${IMAGE_VERSION}

base: ## building base docker image
	docker build --rm -f Dockerfile-base -t vitaflow-base .


build: ## building docker image
	docker build --rm -t vitaflow:${IMAGE_VERSION} .

run: # run the VitaFlow Docker
	@echo "Running vitaflow/vitaflow-app - RUN"
	docker run -d --name vitaflow vitaflow:${IMAGE_VERSION} /bin/bash

rm: ## rm
	docker rm -f vitaflow

exec:
	docker exec -it vitaflow /bin/bash

ps:
	docker ps -a

im:
	# help - https://docs.docker.com/engine/reference/commandline/build/#options
	docker images
