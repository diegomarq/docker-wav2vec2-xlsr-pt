default: build


build:
	docker build --network=host --rm -t w2vs1/wav2vec2-xlsr-s1-train-${USER} .

run: 
	docker run --gpus all --name w2vs1-wav2vec2-xlsr-s1-train-${USER} \
		-it \
		--shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
		-v ${PWD}/homedir/:/root \
		-v ${PWD}/data/:/data \
		-v ${PWD}/models/:/models \
		-v ${PWD}/python/:/xlsr-ft-train \
		w2vs1/wav2vec2-xlsr-s1-train-${USER} \
		bash

stop:
	-docker stop w2vs1-wav2vec2-xlsr-s1-train-${USER}
	-docker rm w2vs1-wav2vec2-xlsr-s1-train-${USER}

clean:
	-docker rmi -f w2vs1/wav2vec2-xlsr-s1-train-${USER}
	sudo rm -rf homedir
	mkdir -p homedir
