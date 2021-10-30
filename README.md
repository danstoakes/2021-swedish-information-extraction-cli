# Swedish Information Extraction Tool
A command line information extraction tool which works with Swedish text. The tool performs various natural language processing tasks, such as parsing, named entity recognition and information extraction. Swedish text can either be input using custom files in the `input_data` directory, or sample data generated from the SUC 3.0 corpus in `training_data`.

This tool was written alongside my undergraduate level dissertation which explored Natural Language Processing with Swedish text.

# Installation
In order to use the tool, various packages need to be installed. It is recommended that the package installer, pip, is used for this process. Installation can be achieved using the following command:

`$ pip install -r requirements.txt`

Once the packages are installed, the tool can be ran using the command:

`$ python3 swedish_information_extraction mode source`

mode can be either: `parse` for the parser module, `ner` for the ner module, or `ie` for the information extraction module.

source can be either a filename, such as `sample_swedish.txt`, which is provided out of the box, or `--sample` which uses excerpts from SUC 3.0. In order to use the SUC 3.0 corpus, it must first be downloaded from [here](https://spraakbanken.gu.se/en/resources/suc3) and placed in the `training_data` folder with the filename "suc3.xml".

# Requirements
The tool requires a model to operate. Models can be created through custom spaCy pipelines or downloaded from the internet. The model which was used during development was the `sv_model_xpos` model available [here](https://spraakbanken.gu.se/en/resources/suc3). There are both UPOS and XPOS-tagged models available, with the XPOS model using Swedish-specific tags, while UPOS uses universal tags. These models have very small differences in performance between them and are both sufficient.

# Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

It is worth noting that this is a legacy-style archive, and likely won't be updated.
