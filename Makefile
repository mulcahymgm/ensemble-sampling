# REDNAFI contact: redowan.nafi@gmail.com
# original - https://gist.github.com/rednafi/7b3b232a2c4b13875b7d6e6d7e0f8d85

#mm 0  modified to suit my environment
# Make sure your environment is named MScAI 		\
													\
Usage (line =black line length, path = action path) \
------												\
make pylinter										\
													\
or,													\
													\
make pylinter line=79 path=myfolder

path := .
line := 88

pylinter:
# raises error if environment is not active
ifneq ("$(CONDA_DEFAULT_ENV)","MScAI")
	@echo "MScAI is not activated!"
	@echo "Activate MScAI first."
	@echo
	exit 1
endif

# checks if black is installed
ifeq ("$(wildcard $(CONDA_PREFIX)/bin/black)","")
	@echo "Installing Black..."
	@pip install black
endif

# checks if isort is installed
ifeq ("$(wildcard $(CONDA_PREFIX)/bin/isort)","")
	@echo "Installing Isort..."
	@pip install isort
endif

# checks if flake8 is installed
ifeq ("$(wildcard $(CONDA_PREFIX)/bin/flake8)","")
	@echo "Installing flake8..."
	@pip install flake8
	@echo
endif

# black
	@echo "Applying Black"
	@echo "---------------\n"
	@black -l $(line) $(path)
	@echo

# isort
	@echo "Applying Isort"
	@echo "---------------\n"
	@isort --atomic --profile black $(path)
	@echo

# flake8
	@echo "Applying Flake8"
	@echo "---------------\n"
	@flake8 --max-line-length "$(line)" \
			--max-complexity "18" \
			--select "B,C,E,F,W,T4,B9" \
			--ignore "E203,E266,E501,W503,F403,F401,E402" \
			--exclude ".tox,.git,__pycache__,old, build, \
						dist, venv"
