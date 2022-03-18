# Cross-Quality Labeled Faces in the Wild (XQLFW)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Downloads](https://img.shields.io/github/downloads/martlgap/xqlfw/total)](https://img.shields.io/github/downloads/martlgap/xqlfw/total)
[![License](https://img.shields.io/badge/license-MIT-blue)](https://img.shields.io/badge/license-MIT-blue)
[![Last Commit](https://img.shields.io/github/last-commit/martlgap/xqlfw)](https://img.shields.io/github/last-commit/martlgap/xqlfw)


Here, we release the database, evaluation protocol and code for the following paper:
- [Cross Quality LFW: A database for Analyzing Cross-Resolution Image Face Recognition in Unconstrained Environments
](https://arxiv.org/pdf/2108.10290.pdf)

## 📂 Database and Evaluation Protocol
If you are interested in our Database and Evaluation Protocol please visit our [website](https://martlgap.github.io/xqlfw).
- [https://martlgap.github.io/xqlfw](https://martlgap.github.io/xqlfw)

## 💻 Code
We provide the code to calculate the accuracy for face recognition models on the XQLFW evaluation protocol. 

### 🥣 Requirements
[![Python 3.8](https://img.shields.io/badge/Python-3.8-blue)](https://img.shields.io/badge/Python-3.8-blue)

### 🚀 How to use
1. Download the database and evaluation protocol [here](https://martlgap.github.io/xqlfw/pages/download)
2. Inference the images and save the embeddings and labels to a numpy file (*.npy) according to: 
    ```python
    [[pair1_img1_embed, pair1_img2_embed, pair2_img1_embed, pair2_img2_embed, ...], 
    [True, True, False, ...]]
    ```
3. Run the evaluate.py code with `--source_embedding` argument 
containing the absolute path to a directory containing your embedding .npy files:
    ```shell
    python evaluate.py --source_embeddings="path/to/your/folder" --csv --save
    ```
    - Use the flag `--csv` if you want to get the results displayed in csv instead of a table.
    - Use the flag `--save` to save the results into the source_embedding directory.
4. See the results and enjoy!

### 📖 Cite
If you use our code please consider citing:
~~~tex
@inproceedings{knoche2021xqlfw,
  author={Knoche, Martin and Hoermann, Stefan and Rigoll, Gerhard},
  booktitle={2021 16th IEEE International Conference on Automatic Face and Gesture Recognition (FG 2021)}, 
  title={Cross-Quality LFW: A Database for Analyzing Cross- Resolution Image Face Recognition in Unconstrained Environments}, 
  year={2021},
  volume={},
  number={},
  pages={1-5},
  doi={10.1109/FG52635.2021.9666960}
}
~~~
and mabybe also:
~~~tex
@TechReport{LFWTech,
  author={Gary B. Huang and Manu Ramesh and Tamara Berg
    and Erik Learned-Miller},
  title={Labeled Faces in the Wild: A Database for Studying
    Face Recognition in Unconstrained Environments},
  institution={University of Massachusetts, Amherst},
  year={2007},
  number={07-49},
  month={October}
}

@TechReport{LFWTechUpdate,
  author={Huang, Gary B and Learned-Miller, Erik},
  title={Labeled Faces in the Wild: Updates and New
    Reporting Procedures},
  institution={University of Massachusetts, Amherst},
  year={2014},
  number={UM-CS-2014-003},
  month={May}
}
~~~

## ✉️ Contact
For any inquiries, please open an [issue](https://github.com/Martlgap/xqlfw/issues) on GitHub or send an E-Mail to: [Martin.Knoche@tum.de](mailto:Martin.Knoche@tum.de)
