# Ibn Bonito

Ibn Bonito is an FPGA based solution to run [Bontio](https://github.com/nanoporetech/bonito) an Oxford Nanopore Reader.

> Since pip install ont-bonito does not work this repo contains the non cli way to do the same.

Get started off by cloning the repo:

```bash
git clone https://github.com/faisalill/ibn-bonito.git 
```

Install all the required python modules:

```bash
pip install -r requirements.txt
```

## download.py

Run

```bash
python download.py
```

To show all available models:

```
download.py -> run() -> download_models_show()
```

To download a specific model:

```
download.py -> run() -> download_model_specific(model_name)
```

> To change the default output directory where models are stored:
> Change
> download.py -> `__models_dir__ = __dir__ / "{desired_directory}"`
> Default is "models"

To download all models:

```
download.py -> run() -> download_models_all(model_list)
```

To show all training data sets:

```
download.py -> run() -> download_training_show_all()
```

To download training data set

```
download.py -> run() -> download_training_data_specific(training_data_set_name)
```
