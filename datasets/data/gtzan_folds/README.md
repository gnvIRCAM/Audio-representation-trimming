We use the folds from [this repository](https://github.com/julianofoleiss/gtzan_sturm_filter_3folds_stratified).

# Stratified 3-fold artist-filtered cross-validation split for the GTZAN Dataset

This repository contains a 3-fold cross-validation split of the GTZAN Dataset. I tried the best I could to follow the recommendations from the paper [The GTZAN dataset: Its contents, its faults, their effects on evaluation, and its future use
](https://arxiv.org/abs/1306.1461) and its companion website: [here](http://www.eecs.qmul.ac.uk/~sturm/research/GTZANtable2/index.html), by Bob Sturm. His recommendations include:

* Removing all exact song duplicates;
* Removing all unrecognizably distorted songs; and
* Applying an artist filter: making sure that, for a given fold, no songs from the same artist are on the training and the test sets at the sime time.

Considering the huge amount of songs from the same artists in all genres, and to keep the folds reasonably stratified, the best I could do was to create a 3-fold split. I also included a jupyter notebook with the code used to generate the splits.

## f*_train.txt files

These files contain the lists of the files that should be used to train the predictive models. The format is the following:

> song_file\\tgenre\\n

## f*_test.txt files

These files contain the lists of files that should be used to test the predictive models. The format is the following:

> song_file\\n

## f*_evaluate.txt files

These files contain the exact same list of files of the correspondent test file, except that they also contain the ground-truth:

> song_file\\tgenre\\n

## f*_*_groups.txt files

Every artist is given an ID during the folder creation process for artist-filtering purposes. I also decided to include these files, which contain the artist ID for every song in the split, in case you want to split each fold further. For example, one might wish to an artist-filtered validation set during model training, thus, it is necessary to know the artist of every song. I have also included a **stratified_group_split** function in the jupyter-notebook that fits this use case.

If you use this split, please cite the following paper:

    @article{FOLEIS2020106127,
        title = "Texture selection for automatic music genre classification",
        journal = "Applied Soft Computing",
        volume = "89",
        pages = "106127",
        year = "2020",
        issn = "1568-4946",
        doi = "https://doi.org/10.1016/j.asoc.2020.106127",
        url = "http://www.sciencedirect.com/science/article/pii/S1568494620300673",
        author = "Juliano Henrique Foleis and Tiago Fernandes Tavares",
        keywords = "Music genre classification, Sound texture selection, Music classification, Signal processing, Music information retrieval",
    }

Thanks to Bob Sturm for the source material used to create this split.

Cheers!

Juliano

Lecturer @ Universidade Tecnologica Federal do Parana (Campo Mourao, PR, Brazil) 

PhD Candidate @ UNICAMP (Campinas, SP, Brazil)

julianofoleiss *at* utfpr *dot* edu *dot* br


