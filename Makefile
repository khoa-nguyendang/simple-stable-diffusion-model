prepare:
	mkdir -p _volumes
	mkdir -p _dataset
	python3 -m venv env
	. env/bin/active
	pip install -r requirements.txt

build:
	docker-compose build .

up:
	docker-compose up -d

down:
	docker-compose down